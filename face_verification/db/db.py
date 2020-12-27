"""
Contains the code related for face recogniction/verification using on a 
database.
"""

from typing import Tuple, Optional, Union
import sqlite3
import numpy as np
import torch


class Database:
    """
    A simple wrapper for sqlite database
    """

    def __init__(self, path):
        self.path = path

    def create(self, table_name="tt") -> None:
        """Create a new table. The table name itself is not very important since we are not doing any relational searches.

        Args:
            table_name (str, optional):  Defaults to "tt".

        Returns:
            None
        """
        query = "CREATE TABLE ? (name text, value blob)"
        q_new = query.replace("?", table_name)
        self._execute(q_new, ())

        return None

    def insert(self, name: str, value: bytes, log: bool = False):
        """Inserts values into the database.

        Args:
            name (str): Name of the person.
            value (bytes): 512 dimensional embeddings as bytes.
            log (bool, optional): Logs to console. Defaults to False.
        """
        self._execute("INSERT INTO tt VALUES (?, ?)",
                      (name, value), fetch=False)

    def __iter__(self):
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute("Select * From tt")
        conn.commit()
        value = c.fetchall()
        conn.close()

        for x in value:
            yield (x)

    def get_value(self, name: str) -> bytes:
        """Returns the embedding associated with the name.

        Args:
            name (str): Name of the person in the database.

        Returns:
            bytes: Bytes type embedding.
        """
        value = self._execute(
            "Select name, value From tt where name=(?)", (name,), fetch=True
        )
        return value

    def _execute(
        self, query, query_params, fetch=False
    ) -> Optional[Union[Tuple[str, float], None]]:
        """Executes the given query with query params, optionally return the output.

        Returns:
            (name, value) or None
        """
        value = None

        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute(query, query_params)
        if fetch:
            value = c.fetchone()

        conn.commit()
        conn.close()

        return value

    def delete(self, name: str) -> None:
        """Deletes the matching name from the database

        Args:
            name (str): name to match
        """
        self._execute("DELETE FROM tt WHERE name = ?", (name,), fetch=False)

    def update(self, name: str, value: bytes) -> None:
        """Updates the embedding associated with the name.

        Args:
            name (str): Name of the person in the database.
            value (bytes): Embedding as bytes
        """
        self._execute(
            "UPDATE tt SET value = ? WHERE name = ?", (value, name), fetch=False
        )

    def drop_table(self, table: str) -> None:
        """Drops the table

        Args:
            table (str): Name of the table.
        """
        query = "DROP TABLE ?".replace("?", table)
        self._execute(query, (), fetch=False)


def torch_to_np(array: torch.Tensor) -> np.ndarray:
    """Coverts torch tensor to numpy array, handles the case when torch tensor is
     stuck in cuda.

    Args:
        array (torch.Tensor): Pytorch Tensor

    Returns:
        np.ndarray
    """
    if array.is_cuda:
        array = array.cpu()
    return array.detach().numpy()


def load_array(byte: bytes) -> np.ndarray:
    """Loads the given bytes array in a ndarray

    Args:
        byte (bytes): Bytes array

    Returns:
        np.ndarray: Numpy array.
    """
    return np.ndarray(shape=(1, 512), dtype=np.float32, buffer=byte)


def compare_embedding(
    embedding: Union[torch.Tensor, np.ndarray], db: Database
) -> Union[Tuple[str, float], None]:
    """Compares the given embedding with every embedding in db. Returns the first
     matched name and distance if any is found.

    Args:
        embedding (Union[torch.Tensor, np.ndarray]): Embedding vector, either numpy or torch
        db (Database): Database to search the images

    Returns:
        Union[Tuple[str, float], None]: Name, distance pair if matched.
    """

    if isinstance(embedding, torch.Tensor):
        embedding = torch_to_np(embedding)
    for name, em in db:
        arr = load_array(em)
        print(arr.shape)
        diff = arr - embedding
        dist = np.sqrt(np.einsum("ij,ij->j", diff, diff))[0]
        print(dist)
        if dist < 0.56:
            return name, dist
    print("Not found")

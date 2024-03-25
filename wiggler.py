"""
Take a connection matrix column vector and convert it into a human readable cp image. 

First, since the vertex positions are discretized to a grid but are not necessarily on the grid, we need to apply a gradient descent to make each vertex as flat foldable as possible.
"""
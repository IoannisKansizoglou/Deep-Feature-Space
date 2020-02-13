#  ==================================================================================
#
#  Copyright (c) 2020, Ioannis Kansizoglou
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
#  ==================================================================================

import numpy as np


class Cl:
    '''
    A Clifford Geometric Algebra in n-Euclidean space, defining a set of 2^n basis elements.
        e.g.    Cl(2) := ['e0', 'e1', 'e2', 'e1e2']
                Cl(3) := ['e0', 'e1', 'e2', 'e3', 'e1e2', 'e1e3', e2e3', 'e1e2e3']

    Parameters
    ----------

    dimensions: int
        The dimensions of the created Euclidean space       
    '''
    def __init__(self, dimensions):

        self.dimensions = int(dimensions)

    def _complete(self,element):
        '''
        Completes the element with zeros from the left to agree with the format of the Algebra.

        Parameters
        ----------

        element: str
            The element as a string of characters

        Returns
        -------

            The element with length equal to the digits of the maximum dimension as type str
        '''      
        return '0'*(len(str(self.dimensions))-len(element))+element

    def _expand2basis(self, element):
        '''
        Expands the element to a list of the sorted basis elements of the Algebra.

        Parameters
        ----------

        element: str
            The element as a string of characters

        Returns
        -------

            The element converted to a list of basis elements as type list()
        '''
        return element.split('e')[1:]

    def _intersection(self, lst1, lst2, sort=True):
        '''
        Calculates the common elements between two lists

        Parameters
        ----------

        lst1: list(), lst2: list()
            The two lists to calculate their intersection.

        sort: Bool
            If true the sorted intersection of two lists is returned. Otherwise the initial order is maintained.

        Returns
        -------

            The (sorted) intersection of the two lists as type of list()
        '''
        lst = list(set(lst1) & set(lst2))
        if sort:
            lst = sorted(lst)
        return lst

    def _union(self, lst1, lst2, sort=True):
        '''
        Calculates the union of two lists without repetition.

        Parameters
        ----------

        lst1: list(), lst2: list()
            The two lists to calculate their union

        sort: Bool
            If true the sorted union of two lists is returned. Otherwise the initial order is maintained.

        Returns
        -------

            The (sorted) union of the two lists as type of list()
        '''
        lst = list(set(lst1) | set(lst2))
        if sort:
            lst = sorted(lst)
        return lst

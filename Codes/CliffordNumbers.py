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
from CliffordSpace import Cl


class ClNumber(Cl):
    '''
    A number of Clifford Algebra named as Cl-number

    Parameters
    ----------

    cl: Cl
        An object of type Cl describing the Clifford Algebra

    coordinates: {'name': float}
        The coordinates of the Cl-number in the orthonormal basis
        e.g.    ClVector(cl2, {'e1': 1.0, 'e2': 2.0})
                ClVector(cl3, {'e1': 1.0, 'e2': 2.0, 'e3': 3.0})
    '''
    def __init__(self, cl, coordinates):

        self.dimensions = cl.dimensions
        self.coordinates = coordinates
        self._discardElements()

    def _discardElements(self,epsilon=1e-10):
        '''
        Discards from the Clifford number all the existing elements with very small coordinates
            e.g. ClNum := {'e1': 1.0, 'e2': 0.0, 'e3': 2.0}
                 ClNum.discard_elements() := {'e1': 1.0, 'e3': 2.0}

        Parameters
        ----------

        epsilon: float
            The smallest absolute value of the kept coordinates

        Returns
        -------
        
            None
        '''
        names = list()
        for name,value in self.coordinates.items():
            
            if abs(value) <= epsilon:
                names.append(name)
        
        for name in names:
            
            self.coordinates.pop(name, None)
        pass

    def _changeSign(self, nameExpanded):

        sortedIndexes = np.array(sorted(range(len(nameExpanded)), key=lambda k: nameExpanded[k]))
        return int(np.sum(np.abs(np.arange(len(sortedIndexes))-sortedIndexes))/2)%2==1

    def _norm(self):
        '''
        Calculates the norm of a Clifford number

        Parameters
        ----------

            None

        Returns
        -------

            The norm of the Clifford number as type of float
        '''
        sq = 0
        for _,value in self.coordinates.items():

            sq += pow(value,2)
        
        return  float(sq**(float(1)/2))

    def _normalize(self):
        '''
        Normalizes a Clifford Blade so as to have norm equal to one

        Parameters
        ----------

            None

        Returns
        -------

            The normalized Clifford number as type of ClNumber
        '''
        norm = self._norm()
        return self.__rmul__(1/norm)


    def __add__(self,cliffordNumber):
        '''
        Calculates the elementwise addition between two Clifford numbers. The elementwise addition is calculated with the addition operator (+) between two ClNumber objects.
            e.g.    ClNum1+ClNum2
        
        Parameters
        ----------

        cliffordNumber: ClNumber
            An object of ClNumber class to add

        Returns
        -------

            The result of the elementwise addition as a new object of type ClNumber
        '''
        resultedCoordinates = self.coordinates.copy()
        
        for name,value in cliffordNumber.coordinates.items():

                if name in resultedCoordinates:
                    resultedCoordinates[name] += value

                else:
                    resultedCoordinates.update({name: value})

        return ClNumber(self, resultedCoordinates)

    def __sub__(self,cliffordNumber):
        '''
        Calculates the elementwise subtraction between two Clifford numbers. The elementwise subtraction is calculated with the subtraction operator (-) between two ClNumber objects.
            e.g.    ClNum1-ClNum2
        
        Parameters
        ----------

        cliffordNumber: ClNumber
            An object of ClNumber class to subtract

        Returns
        -------

            The result of the elementwise subtraction as a new object of type ClNumber
        '''
        resultedCoordinates = self.coordinates.copy()

        for name,value in cliffordNumber.coordinates.items():

                if name in resultedCoordinates:
                    resultedCoordinates[name] -= value

                else:
                    resultedCoordinates.update({name: -value})
                    
        return ClNumber(self, resultedCoordinates)
    
    def __mul__(self,cliffordNumber):
        '''
        Calculates the geometrical product between two Clifford numbers. The geometrical product is calculated using the multiplication operator (*) between two objects of type ClNumber.
            e.g.    clNum1*clNum2

        Parameters
        ----------

        cliffordNumber: ClNumber
            An object of ClNumber class to calculate the geometrical product with

        Returns
        -------

            The result of the geometrical product as a new object of type ClNumber
        '''
        resultedCoordinates = dict()

        for name1,value1 in self.coordinates.items():

            nameExpanded1 = self._expand2basis(name1)

            for name2,value2 in cliffordNumber.coordinates.items():

                nameExpanded2 = self._expand2basis(name2)
                nameExpanded = nameExpanded1 + nameExpanded2

                U = self._union(nameExpanded1, nameExpanded2)
                I = self._intersection(nameExpanded1, nameExpanded2)
                [U.remove(n) for n in I]
                name = sorted([self._complete(n) for n in U]) 
                if len(name) > 0:
                    name = 'e'+'e'.join(name)
                else:
                    name=''
                
                value = value1*value2

                if self._changeSign(nameExpanded):
                    value = -value

                if name in resultedCoordinates:
                    resultedCoordinates[name] += value
                else:
                    resultedCoordinates.update({name: value})

        return ClNumber(self, resultedCoordinates)

    def __pow__(self,cliffordNumber):
        '''
        Calculates the inner (kind of dot) product between two Clifford numbers. The inner product is calculated using the pow operator (**) between two objects of type ClNumber.
            e.g.    clNum1**clNum2

        Parameters
        ----------

        cliffordNumber: ClNumber
            An object of ClNumber class to calculate the inner product with

        Returns
        -------

            The result of the inner product as a new object of type ClNumber
        '''
        resultedCoordinates = dict()

        for name1,value1 in self.coordinates.items():

            nameExpanded1 = self._expand2basis(name1)

            for name2,value2 in cliffordNumber.coordinates.items():

                nameExpanded2 = self._expand2basis(name2)
                
                I = self._intersection(nameExpanded1, nameExpanded2)

                if I == nameExpanded1 or I == nameExpanded2:

                    nameExpanded = nameExpanded1 + nameExpanded2

                    U = self._union(nameExpanded1, nameExpanded2)
                    [U.remove(n) for n in I]
                    name = sorted([self._complete(n) for n in U]) 
                    if len(name) > 0:
                        name = 'e'+'e'.join(name)
                    else:
                        name=''
                        
                    value = value1*value2

                    if self._changeSign(nameExpanded):
                        value = -value

                    if name in resultedCoordinates:
                        resultedCoordinates[name] += value
                    else:
                        resultedCoordinates.update({name: value})

        return ClNumber(self, resultedCoordinates)

    def __xor__(self,cliffordNumber):
        '''
        Calculates the outer (wedge) product between two Clifford numbers. The outer product is calculated using the XOR operator (^) between two objects of type ClNumber.
            e.g.    clNum1^clNum2

        Parameters
        ----------

        cliffordNumber: ClNumber
            An object of ClNumber class to calculate the outer product with

        Returns
        -------

            The result of the outer product as a new object of type ClNumber
        '''
        resultedCoordinates = dict()

        for name1,value1 in self.coordinates.items():

            nameExpanded1 = self._expand2basis(name1)

            for name2,value2 in cliffordNumber.coordinates.items():

                nameExpanded2 = self._expand2basis(name2)
                
                I = self._intersection(nameExpanded1, nameExpanded2)

                if len(I) == 0:

                    nameExpanded = nameExpanded1 + nameExpanded2

                    U = self._union(nameExpanded1, nameExpanded2)
                    [U.remove(n) for n in I]
                    name = sorted([self._complete(n) for n in U]) 
                    if len(name) > 0:
                        name = 'e'+'e'.join(name)
                    else:
                        name=''

                    value = value1*value2

                    if self._changeSign(nameExpanded):
                        value = -value

                    if name in resultedCoordinates:
                        resultedCoordinates[name] += value
                    else:
                        resultedCoordinates.update({name: value})

        return ClNumber(self, resultedCoordinates)

    def __or__(self,cliffordNumber):
        '''
        Calculates the left contraction between two Clifford numbers. The left contraction is calculated using the OR operator (|) between two objects of type ClNumber.
            e.g.    clNum1|clNum2

        Parameters
        ----------

        cliffordNumber: ClNumber
            An object of ClNumber class to calculate the left contraction with

        Returns
        -------

            The result of the left contraction as a new object of type ClNumber
        '''
        resultedCoordinates = dict()

        for name1,value1 in self.coordinates.items():

            nameExpanded1 = self._expand2basis(name1)

            for name2,value2 in cliffordNumber.coordinates.items():

                nameExpanded2 = self._expand2basis(name2)
                
                I = self._intersection(nameExpanded1, nameExpanded2)

                if I == nameExpanded1:

                    nameExpanded = nameExpanded1 + nameExpanded2

                    U = self._union(nameExpanded1, nameExpanded2)
                    [U.remove(n) for n in I]
                    name = sorted([self._complete(n) for n in U]) 
                    if len(name) > 0:
                        name = 'e'+'e'.join(name)
                    else:
                        name=''
                        
                    value = value1*value2

                    if self._changeSign(nameExpanded):
                        value = -value

                    if name in resultedCoordinates:
                        resultedCoordinates[name] += value
                    else:
                        resultedCoordinates.update({name: value})

        return ClNumber(self, resultedCoordinates)

    def __neg__(self):
        '''
        Calculates the negation of a Clifford number. Negation is calculated with the subtraction operator (-) from the left of a ClNumber object.
            e.g.    -clNum

        Parameters
        ----------

            None

        Returns
        -------

            The result of the negation as a new object of type ClNumber
        '''
        resultedCoordinates = dict()
        
        for name,value in self.coordinates.items():
            
            resultedCoordinates.update({name: -value})
            
        return ClNumber(self, resultedCoordinates)
    
    def __rmul__(self,scalar):
        '''
        Multiplies each element of a Clifford number with a scalar value. Scalar multiplication is applied with the multiplication operator (*) from the left of a ClNumber object.
            e.g.    scalar*clNum

        Parameters
        ----------

        scalar: float
            A scalar value to multiply each element of the ClNumber

        Returns
        -------

            The result of the scalar multiplication as a new object of type ClNumber
        '''
        resultedCoordinates = dict()
        
        for name,value in self.coordinates.items():
            
            resultedCoordinates.update({name: scalar*value})
            
        return ClNumber(self, resultedCoordinates)


class ClVector(ClNumber):
    '''
    A vector (1-blade element) of Clifford Algebra named as ClVector

    Parameters
    ----------

    cl: Cl
        An object of Cl describing the Clifford Algebra

    coordinates: numpy.float64
        The coordinates of the Cl-number in the orthonormal basis
        e.g.    ClVector(cl2, [1,2]) => {'e1': 1.0, 'e2': 2.0}
                ClVector(cl3, [1,2,3]) => {'e1': 1.0, 'e2': 2.0, 'e3': 3.0}
                
    Raises
    ------
    
        Error[1]: Number of coordinates more than basis elements of Clifford Algebra
    '''
    def __init__(self, cl, coordinates):

        self.dimensions = cl.dimensions
        self.coordinates = dict()

        if len(coordinates) > cl.dimensions:
            print('ERROR[1]: More values for '+str(cl.dimensions)+' dimensions given!')

        for counter,coord in enumerate(coordinates):
 
            if coord != 0:

                self.coordinates.update({'e'+self._complete(str(counter+1)): float(coord)})

    def _transform2numpy(self):

        vector = np.zeros(self.dimensions)
        
        for name,value in self.coordinates.items():

            vector[int(name[1:])-1] = value

        return vector

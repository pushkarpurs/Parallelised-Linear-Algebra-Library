#ifndef LINALGLIB_H
#define LINALGLIB_H

#include <vector>
#include <memory>
#include <omp.h>

//Important: This implementation assumes the inputs and the returned outputs are of the form of a pointer to a contiguous memory allocation. For example even though int Arr[5][5] is a 2-d array, Arr is a pointer to a contiguous memory allocation and not a double pointer (which we dont want) it can be treated as Arr[25].
//This also means that the number of rows or columns must be provided to the function as this cant be inferred from the passed pointer. 

class LinAlg
{
	private:
	std::vector<std::unique_ptr<double[]>> created_arrays;
	
	public:
	//Template is used here as the function can take either int* or double* as the inputs. 
	template <typename T>
	double* add(const T* a, const T* b, int rows, int cols)
	{
		
		//The below portion will be similar for all functions. Rename the double* sumptr to the name of the operation it is being used for
		auto newArray = std::make_unique<double[]>(rows * cols);
        double* sumptr = newArray.get();
        created_arrays.push_back(std::move(newArray));
		
		//Uncecessary to spawn threads for smaller matrices. Threading overhead dominates and SIMD is better
		//Note SIMD is an intra processor instruction. Bascially a vector instruction, but omp paralell is inter processor spawning multiple threads
		if(rows*cols <=128)
		{
			#pragma omp simd
			for(int i=0; i<rows*cols;i++)
			{
				sumptr[i]=a[i]+b[i];
			}
		}
		else
		{
			#pragma omp parallel for
			for(int i=0; i<rows*cols;i++)
			{
				sumptr[i]=a[i]+b[i];
			}
		}
			
		return sumptr;
	}
	
	template <typename T>
	double dot(const  T* a, const T* b, int rows, int cols)
	{	
		if(cols!=1)
		{
			std::cout<<"ERROR: Dot product can only be computed for vectors. Matrices unsupported. Use matmul"<<std::endl;
			return NaN;
		}
		double s=0.0;
		if(rows<=128)
		{
			#pragma omp simd
			for(int i=0; i<rows; i++)
			{
				s+=a[i]*b[i];
			}
		}
		else
		{
			#pragma omp parallel for reduction(+:s)
			for(int i=0; i<rows; i++)
			{
				s+=a[i]*b[i];
			}
		}
		return s;
	}
	
	template <typename T>
	double* matmul(const  T* a, const T* b, int rowsa, int colsa, int rowsb, int colsb)
	{
		if(colsa!=rowsb)
		{
			std::cout<<"Error: Mismatch of matrix dimentions"<<std::endl;
			return nullptr;
		}
		auto newArray = std::make_unique<double[]>(rows * cols);
        double* matmul = newArray.get();
        created_arrays.push_back(std::move(newArray));
	}

	
    	// Matrix multiplication
	template <typename T>
	double* multiply(const T* a, const T* b, int rows, int cols) 
	{
		auto newArray = std::make_unique<double[]>(rows * cols);
		double* resultptr = newArray.get();
		
		if(rows <= 64) 
		{
		    	for(int i = 0; i < rows; i++) 
			{
				for(int j = 0; j < cols; j++) 
				{
			    		double sum = 0;
			    		#pragma omp simd reduction(+:sum)
			    		for(int k = 0; k < cols; k++) 
					{
						sum += a[i*cols + k] * b[k*cols + j];
			    		}
			    		resultptr[i*cols + j] = sum;
				}
		    	}
		}
		else 
		{
		    	//collapse(2) only if the number of threads is more than the number of rows in the matrix and for matrices of size >= 64
		    	#pragma omp parallel for collapse(2) 
		    	for(int i = 0; i < rows; i++) 
			{
				for(int j = 0; j < cols; j++) 
				{
			    		double sum = 0;
			    		#pragma omp simd reduction(+:sum)
			    		for(int k = 0; k < cols; k++) 
					{
						sum += a[i*cols + k] * b[k*cols + j];
			    		}
			    		resultptr[i*cols + j] = sum;
				}
		    	}
		}
		
		created_arrays.push_back(std::move(newArray));
		return resultptr;
	}
	    
    	// Creating identity matrix for base case (exponent == 0)
    	double* identity(int size) 
	{
		auto newArray = std::make_unique<double[]>(rows * cols);
		double* idptr = newArray.get();
		if (size <= 64) 
		{  
	    		#pragma omp simd
	    		for (int i = 0; i < size * size; i++) 
			{
				idptr[i] = (i / size == i % size) ? 1.0 : 0.0;
	    		}
		} 
		else 
		{  
	    		#pragma omp parallel for
	    		for (int i = 0; i < size * size; i++) 
			{
				idptr[i] = (i / size == i % size) ? 1.0 : 0.0;
	    		}
		}
	
		return idptr;
	}
    
    
   	 // Using binary exponentiation for reducing number of matrix multiplications
	template <typename T>
	double* power(const T* matrix, int size, int exponent) 
	{
		if(exponent == 0) 
		{
			double* idMatrix = identity(size); 
		   	created_arrays.push_back(std::unique_ptr<double[]>(idMatrix)); 
		   	return idMatrix;
		}
		
		if(exponent == 1) 
		{
		    	double* originalMatrix = matrix;          
		    	created_arrays.push_back(std::unique_ptr<double[]>(originalMatrix));
		    	return originalMatrix;
		}
		
		if(exponent % 2 == 0) 
		{
		   	double* half = power(matrix, size, exponent / 2);
		   	return multiply(half, half, size, size);
			delete[] half;
		} 
		else 
		{
		    	double* powMinusOne = power(matrix, size, exponent - 1);
		    	return multiply(matrix, powMinusOne, size, size);
			delete[] powMinusOne;
		}
	}

    
};

#endif

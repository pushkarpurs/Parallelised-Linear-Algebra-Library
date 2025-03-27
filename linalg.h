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
		double s=0.0;
		if(rows*cols <=128)
		{
			#pragma omp simd
			for(int i=0; i<rows*cols; i++)
			{
				s+=a[i]*b[i];
			}
		}
		else
		{
			#pragma omp parallel for reduction(+:s)
			for(int i=0; i<rows*cols; i++)
			{
				s+=a[i]*b[i];
			}
		}
		return s;
	}
};

#endif
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
	template <typename T, std::size_t cols>
	double (*add(const T (*a)[cols], const T (*b)[cols], int rows))[cols] //function does not get the dimentions of the two matrices seperately and thus does not check for compatibility. It is up to the user to ensure the two matrices are of the same dimentions.
	{
		
		//The below portion will be similar for all functions. Rename the double* sumptr to the name of the operation it is being used for
		auto newArray = std::make_unique<double[]>(rows * cols);
        double* res = newArray.get();
        created_arrays.push_back(std::move(newArray));
		
		double (*sumptr)[cols] = reinterpret_cast<double (*)[cols]>(res);
		
		//Uncecessary to spawn threads for smaller matrices. Threading overhead dominates and SIMD is better
		//Note SIMD is an intra processor instruction. Bascially a vector instruction, but omp paralell is inter processor spawning multiple threads
		
		if(rows*cols <=256)
		{
			for(int i=0; i<rows;i++)
			{
				#pragma omp simd
				for(int j=0;j<cols;j++)
				{
					sumptr[i][j]=a[i][j]+b[i][j];
				}
			}
		}
		else
		{
			#pragma omp parallel for collapse(2) schedule(static)
			for(int i=0; i<rows*cols;i++)
			{
				for(int j=0; j<cols; j++)
				{
					sumptr[i][j]=a[i][j]+b[i][j];
				}
			}
		}
		return sumptr;
	}
	
	//This function is only for vectors. The library does not handle complex numbers and thus does not have different functins for matrix and vector dot products
	template <typename T>
	double dot(const  T* a, const T* b, int rows)
	{	
		double s=0.0;
		if(rows<=256)
		{
			#pragma omp simd
			for(int i=0; i<rows; i++)
			{
				s+=a[i]*b[i];
			}
		}
		else
		{
			#pragma omp parallel for reduction(+:s) schedule(static)
			for(int i=0; i<rows; i++)
			{
				s+=a[i]*b[i];
			}
		}
		return s;
	}
	
	template <typename T, std::size_t colsa, std::size_t colsb>
	double	(*multiply(const T (*a)[colsa], const T (*b)[colsb], int rowsa, int rowsb))[colsb] 
	{
		auto newArray = std::make_unique<double[]>(rowsa * colsb);
		double* res = newArray.get();
		created_arrays.push_back(std::move(newArray));
		
		double (*resultptr)[colsb] = reinterpret_cast<double (*)[colsb]>(res);
		
		if(rowsa*colsb <= 64) 
		{
		    for(int i = 0; i < rowsa; i++) 
			{
				for(int j = 0; j < colsb; j++) 
				{
					#pragma omp simd
					for(int k = 0; k < colsa; k++) 
					{
						resultptr[i][j]+=a[i][k]*b[k][j];
					}
				}
		    }
		}
		else 
		{
		    #pragma omp parallel for collapse(2) schedule(static)
		   	for(int i = 0; i < rowsa; i++) 
			{
				for(int j = 0; j < colsb; j++) 
				{
		    		#pragma omp simd
		    		for(int k = 0; k < colsa; k++) 
					{
						resultptr[i][j]+=a[i][k]*b[k][j];
		    		}
				}
		   	}
		}
		return resultptr;
	}
	
	template <std::size_t cols>
	double (*identity())[cols] 
	{
		auto newArray = std::make_unique<double[]>(cols*cols);
		double* res = newArray.get();
		created_arrays.push_back(std::move(newArray));
		
		double (*idptr)[cols] = reinterpret_cast<double (*)[cols]>(res);
		
		if (cols <= 16) 
		{  
			for (int i = 0; i < cols; i++) 
			{
				#pragma omp simd
				for(int j=0; j< cols; j++)
				{
					if(i==j)
						idptr[i][j]=1.0;
					else
						idptr[i][j]=0.0;
				}
			}
		} 
		else 
		{  
			#pragma omp parallel for schedule(static)
			for (int i = 0; i < cols; i++) 
			{
				for(int j=0; j< cols; j++)
				{
					if(i==j)
						idptr[i][j]=1.0;
					else
						idptr[i][j]=0.0;
				}
			}
		}
	
		return idptr;
	}
	
	template <typename T, std::size_t cols>
	double (*copy(const T (*a)[cols], int rows))[cols]
	{
		auto newArray = std::make_unique<double[]>(rows * cols);
        double* res = newArray.get();
        created_arrays.push_back(std::move(newArray));
		double (*cpy)[cols] = reinterpret_cast<double (*)[cols]>(res);
		
		if(rows*cols <=256)
		{
			for(int i=0; i<rows;i++)
			{
				#pragma omp simd
				for(int j=0;j<cols;j++)
				{
					cpy[i][j]=a[i][j];
				}
			}
		}
		else
		{
			#pragma omp parallel for collapse(2) schedule(static)
			for(int i=0; i<rows*cols;i++)
			{
				for(int j=0; j<cols; j++)
				{
					cpy[i][j]=a[i][j];
				}
			}
		}
		return cpy;
	}
	
	template <typename T, std::size_t cols>
	double (*power(const T (*matrix)[cols], int exponent))[cols] 
	{
		if(exponent == 0) 
		{
			double (*idMatrix)[cols] = identity<cols>(); 
		   	return idMatrix;
		}
		if(exponent == 1) 
		{
		    return copy(matrix,cols);
		}
		if(exponent == 2)
		{
			return multiply(matrix,matrix,cols,cols);
		}
		if(exponent % 2 == 0) 
		{
		   	double (*half)[cols] = power(matrix, exponent / 2);
		   	double (*ret)[cols] = multiply(half, half, cols, cols);
			//deallocating the second last element and removing its entry in created arrays
			created_arrays.erase(created_arrays.end() - 2);
			return ret;
		} 
		else 
		{
		    double (*powMinusOne)[cols] = power(matrix, exponent - 1);
			double (*matcpy)[cols]=copy(matrix,cols);
		    double (*ret)[cols]=multiply(powMinusOne, matcpy, cols, cols);
			created_arrays.erase(created_arrays.end() - 2);
			created_arrays.erase(created_arrays.end() - 2);
			return ret;
		}
	}

	template <std::size_t rows, typename T>
	double (*outer(const T* a,const T* b))[rows]
	{
		auto newArray = std::make_unique<double[]>(rows * rows);
		double* res = newArray.get();
		created_arrays.push_back(std::move(newArray));
		
		double (*opr)[rows] = reinterpret_cast<double (*)[rows]>(res);
		
		if(rows*rows <= 256) 
		{
		    for(int i = 0; i < rows; i++) 
			{
				#pragma omp simd
				for(int j = 0; j < rows; j++) 
				{
					opr[i][j]=a[i]*b[j];
				}
		    }
		}
		else 
		{
		    #pragma omp parallel for collapse(2) schedule(static)
		   	for(int i = 0; i < rows; i++) 
			{
				for(int j = 0; j < rows; j++) 
				{
					opr[i][j]=a[i]*b[j];
				}
		    }
		}
		return opr;
	}
	
	template <std::size_t rows,typename T, std::size_t cols>
	double (*transpose(const T (*a)[cols]))[rows]
	{
		auto newArray = std::make_unique<double[]>(rows * cols);
        double* res = newArray.get();
        created_arrays.push_back(std::move(newArray));
		double (*transp)[cols] = reinterpret_cast<double (*)[cols]>(res);
		
		if(rows*cols <=256)
		{
			for(int i=0; i<rows;i++)
			{
				#pragma omp simd
				for(int j=0;j<cols;j++)
				{
					transp[j][i]=a[i][j];
				}
			}
		}
		else
		{
			#pragma omp parallel for collapse(2) schedule(static)
			for(int i=0; i<rows*cols;i++)
			{
				for(int j=0; j<cols; j++)
				{
					transp[j][i]=a[i][j];
				}
			}
		}
		return transp;
	}
};
#endif

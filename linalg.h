#ifndef LINALGLIB_H
#define LINALGLIB_H

#include <vector>
#include <memory>
#include <omp.h>
#include <cmath>
#include <iostream>
#include <atomic>


//the number of rows or columns must be provided (depends on the function requirements) as this cant be inferred from the passed pointer. 

class LinAlg
{
	private:
	std::vector<std::unique_ptr<double[]>> created_arrays;
	
	public:
	//Template is used here as the function can take either int* or double* as the inputs. 
	template <typename T1, typename T2, std::size_t cols>
	double (*add(const T1 (*a)[cols], const T2 (*b)[cols], int rows))[cols] //function does not get the dimentions of the two matrices seperately and thus does not check for compatibility. It is up to the user to ensure the two matrices are of the same dimentions.
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
			#pragma omp parallel for schedule(static)
			for(int i=0; i<rows;i++)
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
	template <typename T1, typename T2>
	double dot(const  T1* a, const T2* b, int rows)
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
	
	template <typename T1, typename T2, std::size_t colsa, std::size_t colsb>
	double	(*multiply(const T1 (*a)[colsa], const T2 (*b)[colsb], int rowsa, int rowsb))[colsb] 
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
			#pragma omp parallel for schedule(static)
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
		    double (*ret)[cols]=multiply(powMinusOne, matrix, cols, cols);
			created_arrays.erase(created_arrays.end() - 2);
			return ret;
		}
	}

	template <std::size_t rows, typename T1, typename T2>
	double (*outer(const T1* a,const T2* b))[rows]
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
		    #pragma omp parallel for schedule(static)
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
			#pragma omp parallel for schedule(static)
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
	
	template <typename T1, typename T2, std::size_t cols>
	double *vecmat(const T1* a, const T2 (*b)[cols], int rows)
	{
		auto newArray = std::make_unique<double[]>(cols);
        double* vecm = newArray.get();
        created_arrays.push_back(std::move(newArray));
		
		if(rows*cols<256)
		{
			for(int j=0; j<rows; j++)
			{
				#pragma omp simd
				for(int i=0; i<cols;i++)
				{
					vecm[i]+=a[j]*b[j][i];
				}
			}
		}
		else
		{
			#pragma omp parallel for schedule(static)
			for(int j=0; j<rows; j++)
			{
				for(int i=0; i<cols;i++)
				{
					vecm[i]+=a[j]*b[j][i];
				}
			}
		}
		return vecm;
	}
	
	template <typename T1, std::size_t cols, typename T2>
	double *matvec(const T1 (*a)[cols], const T2* b, int rows)
	{
		auto newArray = std::make_unique<double[]>(rows);
        double* matv = newArray.get();
        created_arrays.push_back(std::move(newArray));
		
		if(rows*cols<256)
		{
			for(int j=0; j<rows; j++)
			{
				#pragma omp simd
				for(int i=0; i<cols;i++)
				{
					matv[j]+=b[i]*a[j][i];
				}
			}
		}
		else
		{
			#pragma omp parallel for schedule(static)
			for(int j=0; j<rows; j++)
			{
				for(int i=0; i<cols;i++)
				{
					matv[j]+=b[i]*a[j][i];
				}
			}
		}
		return matv;
	}
	
	//Assumes passed matrix is square
	template <typename T, std::size_t cols>
	double (*cholesky(const T (*a)[cols]))[cols]
	{
		std::atomic<bool> r{false};
		#pragma omp parallel for schedule(static) shared(r)
		for(int i=0; i<cols; i++)
		{
			if(r.load())
				continue;
			for(int j=i+1; j<cols; j++)
			{
				if(a[i][j]!=a[j][i])
				{
					r.store(true);
				}
			}
		}
		if(r.load())
		{
			std::cout<<"Error: The matrix is not symmetric"<<std::endl;
			return nullptr;
		}
		r.store(false);
		auto newArray = std::make_unique<double[]>(cols*cols);
		std::fill_n(newArray.get(), cols * cols, 0.0);
        double* res = newArray.get();
		double (*L)[cols] = reinterpret_cast<double (*)[cols]>(res);
		
		for (int i=0; i<cols; i++) 
		{
			double sum= 0.0;
			for (int k=0;k<i; k++)
			{
				sum+=L[i][k]*L[i][k];
			}
			if(a[i][i]-sum<0)
			{
				std::cout<<"Error: The matrix is not positive definite"<<std::endl;
				return nullptr;
			}
			L[i][i]=std::sqrt(a[i][i]-sum);

			#pragma omp parallel for
			for (int j=i+1; j<cols; j++) 
			{
				if(r.load())
					continue;
				double sum = 0.0;
				for (int k = 0; k< i; k++)
				{
					sum+= L[j][k]*L[i][k];
				}
				if(L[i][i]==0)
				{
					r.store(true);
					continue;
				}
				L[j][i] =(a[j][i]-sum)/L[i][i];
			}
			
			if(r.load())
			{
				std::cout<<"Error: The matrix is not positive definite"<<std::endl;
				return nullptr;
			}
		}
		created_arrays.push_back(std::move(newArray));
		return L;
	}
	
	template <typename T, size_t cols>
	double determinant(const T (*a)[cols], int rows) 
	{
		//kindof implementinf LU here. Though not really calculating L explicitly since we just need the diagonal elements of U
		double lu[rows][cols];
		int perm[rows];
		int swaps = 0;

		#pragma omp parallel for collapse(2)
		for (int i=0; i<rows; i++)
			for (int j=0; j<rows; j++)
				lu[i][j] = a[i][j];

		for (int k=0; k<rows; k++) 
		{
			int pivot = k;
			double maxv = std::abs(lu[k][k]);
			for (int i =k+1; i<rows; i++) 
			{
				if (std::abs(lu[i][k]) > maxv) 
				{
					maxv = std::abs(lu[i][k]);
					pivot = i;
				}
			}

			if (maxv < 1e-12) 
				return 0.0;

			if (pivot != k) 
			{
				std::swap_ranges(lu[k], lu[k]+rows, lu[pivot]);
				swaps++;
			}

			double pivot_val = lu[k][k];

			#pragma omp parallel for
			for (int i = k+1; i<rows; i++) 
			{
				lu[i][k] /= pivot_val;
				for (int j = k + 1; j<rows; j++) 
				{
					lu[i][j] -= lu[i][k] * lu[k][j];
				}
			}
		}
		double det=1.0;
		#pragma omp paralell for schedule(static) reduction(*:det)
		for (int i = 0; i<rows; i++) 
		{
			det *= lu[i][i];
		}
		if(swaps%2==1)
			det*=-1;
		return det;
	}
};
#endif

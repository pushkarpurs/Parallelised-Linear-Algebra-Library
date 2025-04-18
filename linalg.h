#ifndef LINALGLIB_H
#define LINALGLIB_H

#include <vector>
#include <memory>
#include <omp.h>
#include <cmath>
#include <iostream>
#include <atomic>
#include <utility>
#include <algorithm>
#include <cstring> 


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
				#pragma omp simd
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
			// Note: Generally due to the if condition above these matrices are large enough not to require a seperate accumulator variable to prevent false sharing
			#pragma omp parallel for schedule(static)
			for(int i=0; i<rows;i++)
			{
				#pragma omp simd
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
				#pragma omp simd
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
		
		int blockSize = 32; // Tune this depending on your cache size

		if (rows * cols <= 256) {
			for (int i = 0; i < rows; i++) {
				#pragma omp simd
				for (int j = 0; j < cols; j++) {
					transp[j][i] = a[i][j];
				}
			}
		} 
		else {
			#pragma omp parallel for schedule(static)
			for (int ii = 0; ii < rows; ii += blockSize) {
				for (int jj = 0; jj < cols; jj += blockSize) {
					for (int i = ii; i < ii + blockSize && i < rows; i++) {
						#pragma omp simd
						for (int j = jj; j < std::min(jj + blockSize, static_cast<int>(cols)); j++)
						{
							transp[j][i] = a[i][j];
						}
					}
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
		
		if (rows * cols < 256)
		{
			for (int j = 0; j < rows; j++)
			{
				#pragma omp simd
				for (int i = 0; i < cols; i++)
				{
					vecm[i] += a[j] * b[j][i];
				}
			}
		}
		else
		{
			int num_threads = omp_get_max_threads();
			std::vector<std::vector<double>> local_sums(num_threads, std::vector<double>(cols, 0.0));

			#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				std::vector<double>& local_vec = local_sums[tid];

				#pragma omp for schedule(static)
				for (int j = 0; j < rows; j++)
				{
					for (int i = 0; i < cols; i++)
					{
						local_vec[i] += a[j] * b[j][i];
					}
				}
			}

			// Reduction
			for (int t = 0; t < num_threads; t++)
			{
				for (int i = 0; i < cols; i++)
				{
					vecm[i] += local_sums[t][i];
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
		
		if (rows * cols < 256)
		{
			for (int j = 0; j < rows; j++)
			{
				#pragma omp simd
				for (int i = 0; i < cols; i++)
				{
					matv[j] += b[i] * a[j][i];
				}
			}
		}
		else
		{
			std::vector<double> local_sum(rows, 0.0);

			#pragma omp parallel
			{
				std::vector<double> thread_private(rows, 0.0);  // thread-local storage

				#pragma omp for schedule(static)
				for (int j = 0; j < rows; j++)
				{
					double sum = 0.0;
					for (int i = 0; i < cols; i++)
					{
						sum += b[i] * a[j][i];
					}
					thread_private[j] = sum;
				}

				// Reduction step: each thread adds its contribution
				#pragma omp critical
				{
					for (int j = 0; j < rows; j++)
					{
						matv[j] += thread_private[j];
					}
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
	double (*ludecompose(const T (*a)[cols], int rows, int& swaps, int* pivot))[cols] 
	{
			auto newArray = std::make_unique<double[]>(rows * cols);
    		// auto newArray = make_unique_array<double>(rows * cols);
    		double* res = newArray.get();
    		created_arrays.push_back(std::move(newArray));
    		double (*lu)[cols] = reinterpret_cast<double (*)[cols]>(res);

    		for (int i = 0; i < rows; ++i) pivot[i] = i;

   	 	#pragma omp parallel for
    		for (int i = 0; i < rows; i++)
		{
        		for (int j = 0; j < cols; j++)
			{
            			lu[i][j] = a[i][j];
			}
		}

    		swaps = 0;

    		for (int k = 0; k < rows; k++) 
		{
        		int pivot_index = k;
        		double maxv = std::abs(lu[k][k]);
        		for (int i = k + 1; i < rows; i++) 
			{
            			if (std::abs(lu[i][k]) > maxv) 
				{
                			maxv = std::abs(lu[i][k]);
                			pivot_index = i;
            			}
        		}

       			if (maxv < 1e-12)
            			return nullptr;

       			if (pivot_index != k) 
			{
            			std::swap_ranges(lu[k], lu[k] + cols, lu[pivot_index]);
	    			std::swap(pivot[k], pivot[pivot_index]);
            			swaps++;
        		}

        		double pivot_val = lu[k][k];
        		#pragma omp parallel for
        		for (int i = k + 1; i < rows; i++) 
			{
            			lu[i][k] /= pivot_val;
            			for (int j = k + 1; j < cols; j++) 
				{
                			lu[i][j] -= lu[i][k] * lu[k][j];
            			}
        		}
    		}

    		return lu;
	}

	template <typename T, size_t cols>
	double determinant(const T (*a)[cols], int rows) 
	{
    		int swaps = 0;
    		int pivot[rows];
    		double (*lu)[cols] = ludecompose(a, rows, swaps, pivot);
    		if (!lu) 
		{
        		std::cerr << "Matrix is singular.\n";
        		return 0.0;
    		}

    		double det = 1.0;
    		#pragma omp parallel for reduction(* : det)
    		for (int i = 0; i < rows; i++) 
		{
        		det *= lu[i][i];
    		}
    		if (swaps % 2 != 0)
        		det *= -1;
    		return det;
	}

	template <typename T, size_t cols>
	void forwardSubstitution(const T (*lu)[cols], double* y, int rows, int col, const int* pivot) 
	{
    		for (int i = 0; i < rows; ++i) 
		{
			if(pivot[i] == col)
				y[i] = 1.0;
			else
				y[i] = 0.0;
        		for (int j = 0; j < i; ++j)
            			y[i] -= lu[i][j] * y[j];
    		}
	}

	template <typename T, size_t cols>
	void backwardSubstitution(const T (*lu)[cols], double* y, double* x, int rows) 
	{
    		for (int i = rows - 1; i >= 0; --i) 
		{
        		x[i] = y[i];
        		for (int j = i + 1; j < rows; ++j)
            			x[i] -= lu[i][j] * x[j];
       			x[i] /= lu[i][i];
    		}
	}

	template <typename T, size_t cols>
	double (*inverse(const T (*a)[cols], int rows))[cols] 
	{
    		int swaps = 0;
    		int pivot[rows];
    		double (*lu)[cols] = ludecompose(a, rows, swaps, pivot);
    		if (!lu) 
			return nullptr;
			
			auto newArray = std::make_unique<double[]>(rows *cols);
    		//auto newArray = make_unique_array<double>(rows * cols);
    		double* res = newArray.get();
    		created_arrays.push_back(std::move(newArray));
    		double (*inv)[cols] = reinterpret_cast<double (*)[cols]>(res);

    		#pragma omp parallel for
    		for (int col = 0; col < cols; ++col)
		{
        		double y[rows], x[rows];
        		forwardSubstitution(lu, y, rows, col, pivot);
        		backwardSubstitution(lu, y, x, rows);
        		for (int i = 0; i < rows; ++i)
            			inv[i][col] = x[i];
    		}

    		return inv;
	}

	template <typename T, size_t cols>
	std::pair<T*, T*>  qrdecomp(const  T (*a)[cols], int rows)
	{
		auto newArray1 = std::make_unique<double[]>(rows * cols);
    		auto newArray2 = std::make_unique<double[]>(rows * cols);
    		double* q = newArray1.get();
    		double* r = newArray2.get();
    		created_arrays.push_back(std::move(newArray1));
    		created_arrays.push_back(std::move(newArray2));

    		double (*Q)[cols] = reinterpret_cast<double (*)[cols]>(q);
    		double (*R)[cols] = reinterpret_cast<double (*)[cols]>(r);

    		for (int k = 0; k < cols; ++k) 
		{
		        #pragma omp parallel for
        		for (int i = 0; i < rows; ++i)
            			Q[i][k] = a[i][k];

		        for (int j = 0; j < k; ++j) 
			{
				double A_col[rows], Q_col[rows];
            			#pragma omp simd
            			for (int i = 0; i < rows; ++i) 
				{
                			A_col[i] = a[i][k];
                			Q_col[i] = Q[i][j];
            			}
            			double rjk = dot(Q_col,A_col, rows);
            			R[j][k] = rjk;

            			#pragma omp parallel for
            			for (int i = 0; i < rows; ++i)
                			Q[i][k] -= rjk * Q[i][j];
        		}

        		double norm = 0.0;
        		#pragma omp simd reduction(+:norm)
        		for (int i = 0; i < rows; ++i)
            			norm += Q[i][k] * Q[i][k];

        		R[k][k] = std::sqrt(norm);

        		#pragma omp parallel for
		        for (int i = 0; i < rows; ++i)
            			Q[i][k] /= R[k][k];
    		}
		
    		return {reinterpret_cast<T*>(Q), reinterpret_cast<T*>(R)};
	}

	template <typename T, size_t cols>
	bool is_upper_triangular(const T (*A)[cols], double tolerance = 1e-10) 
	{
		bool is_upper = true; 
		#pragma omp parallel for reduction(&:is_upper)
    		for (int i = 1; i < cols; ++i) 
		{
        		for (int j = 0; j < i; ++j) 
			{
            			if (std::abs(A[i][j]) > tolerance) 
				{
                			is_upper = false;
            			}
        		}
    		}
    		return is_upper;
	}

	template <typename T, size_t cols>
	std::vector<double> eigenval(const T (*A)[cols],int rows,int max_iterations = 100, double tolerance = 1e-10) 
	{
    		double (*A_k)[cols]=copy(A, rows);
    
    		for (int iter = 0; iter < max_iterations; ++iter) 
		{
        		if (is_upper_triangular(A_k, tolerance)) 
			{
            			break;
        		}
        
        		auto result = qrdecomp(A_k, rows);
        		double* Q_raw = result.first;
        		double* R_raw = result.second;
        
        		double (*Q)[cols] = reinterpret_cast<double (*)[cols]>(Q_raw);
        		double (*R)[cols] = reinterpret_cast<double (*)[cols]>(R_raw);

			//Not sure about the mem mgmt happening here
        		double (*new_A)[cols]=multiply(R, Q, rows, rows);
        		A_k = copy(new_A, rows);
        
    		}

    		std::vector<double> eigenvalues(rows);
    		if(rows<=16)
		{
        		#pragma omp simd
    			for (size_t i = 0; i < rows; ++i) 
			{
        			eigenvalues[i]=A_k[i][i];
    			}
		}
		else
		{
    			#pragma omp parallel for schedule(static)
    			for (size_t i = 0; i < rows; ++i) 
			{
        			eigenvalues[i]=A_k[i][i];
    			}
		}
		return eigenvalues;
	}

};
#endif

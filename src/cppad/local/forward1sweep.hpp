# ifndef CPPAD_LOCAL_FORWARD1SWEEP_HPP
# define CPPAD_LOCAL_FORWARD1SWEEP_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
\file forward1sweep.hpp
Compute one Taylor coefficient for each order requested.
*/

/*
\def CPPAD_ATOMIC_CALL
This avoids warnings when NDEBUG is defined and user_ok is not used.
If NDEBUG is defined, this resolves to
\code
	user_atom->forward
\endcode
otherwise, it respolves to
\code
	user_ok = user_atom->forward
\endcode
This macro is undefined at the end of this file to facillitate its
use with a different definition in other files.
*/
# ifdef NDEBUG
# define CPPAD_ATOMIC_CALL user_atom->forward
# else
# define CPPAD_ATOMIC_CALL user_ok = user_atom->forward
# endif

/*!
\def CPPAD_FORWARD1SWEEP_TRACE
This value is either zero or one.
Zero is the normal operational value.
If it is one, a trace of every forward1sweep computation is printed.
*/
# define CPPAD_FORWARD1SWEEP_TRACE 0

/*!
Compute arbitrary order forward mode Taylor coefficients.

<!-- replace forward0sweep_doc_define -->
\tparam Base
The type used during the forward mode computations; i.e., the corresponding
recording of operations used the type AD<Base>.

\param s_out
Is the stream where output corresponding to PriOp operations will
be written.

\param print
If print is false,
suppress the output that is otherwise generated by the c PriOp instructions.

\param n
is the number of independent variables on the tape.

\param numvar
is the total number of variables on the tape.
This is also equal to the number of rows in the matrix taylor; i.e.,
play->num_var_rec().

\param play
The information stored in play
is a recording of the operations corresponding to the function
\f[
	F : {\bf R}^n \rightarrow {\bf R}^m
\f]
where \f$ n \f$ is the number of independent variables and
\f$ m \f$ is the number of dependent variables.

\param J
Is the number of columns in the coefficient matrix taylor.
This must be greater than or equal one.

<!-- end forward0sweep_doc_define -->

\param cskip_op
Is a vector with size play->num_op_rec().
\n
\n
<tt>p = 0</tt>
\n
In this case,
the input value of the elements does not matter.
Upon return, if cskip_op[i] is true, the operator with index i
does not affect any of the dependent variable
(given the value of the independent variables).
\n
\n
<tt>p > 0</tt>
\n
In this case cskip_op is not modified and has the same meaning
as its return value above.

\param var_by_load_op
is a vector with size play->num_load_op_rec().
\n
\n
<tt>p == 0</tt>
\n
In this case,
The input value of the elements does not matter.
Upon return,
it is the variable index corresponding the result for each load operator.
In the case where the index is zero,
the load operator results in a parameter (not a variable).
Note that the is no variable with index zero on the tape.
\n
\n
<tt>p > 0</tt>
\n
In this case var_by_load_op is not modified and has the meaning
as its return value above.

\param p
is the lowest order of the Taylor coefficients
that are computed during this call.

\param q
is the highest order of the Taylor coefficients
that are computed during this call.

\param taylor
\n
\b Input:
For <code>i = 1 , ... , numvar-1</code>,
<code>k = 0 , ... , p-1</code>,
<code>taylor[ J*i + k]</code>
is the k-th order Taylor coefficient corresponding to
the i-th variable.
\n
\n
\b Input:
For <code>i = 1 , ... , n</code>,
<code>k = p , ... , q</code>,
<code>taylor[ J*j + k]</code>
is the k-th order Taylor coefficient corresponding to
the i-th variable
(these are the independent varaibles).
\n
\n
\b Output:
For <code>i = n+1 , ... , numvar-1</code>, and
<code>k = 0 , ... , p-1</code>,
<code>taylor[ J*i + k]</code>
is the k-th order Taylor coefficient corresponding to
the i-th variable.


\param compare_change_count
Is the count value for changing number and op_index during
zero order foward mode.

\param compare_change_number
If p is non-zero, this value is not changed, otherwise:
If compare_change_count is zero, this value is set to zero, otherwise:
this value is set to the number of comparision operations
that have a different result from when the information in
play was recorded.

\param compare_change_op_index
if p is non-zero, this value is not changed, otherwise:
If compare_change_count is zero, this value is set to zero.
Otherwise it is the operator index (see forward_next) for the count-th
comparision operation that has a different result from when the information in
play was recorded.
*/

template <class Base>
void forward1sweep(
	const local::player<Base>* play,
	std::ostream&              s_out,
	const bool                 print,
	const size_t               p,
	const size_t               q,
	const size_t               n,
	const size_t               numvar,
	const size_t               J,
	Base*                      taylor,
	bool*                      cskip_op,
	pod_vector<addr_t>&        var_by_load_op,
	size_t                     compare_change_count,
	size_t&                    compare_change_number,
	size_t&                    compare_change_op_index
)
{
	// number of directions
	const size_t r = 1;

	CPPAD_ASSERT_UNKNOWN( p <= q );
	CPPAD_ASSERT_UNKNOWN( J >= q + 1 );
	CPPAD_ASSERT_UNKNOWN( play->num_var_rec() == numvar );

	/*
	<!-- replace forward0sweep_code_define -->
	*/
	// op code for current instruction
	OpCode op;

	// index for current instruction
	size_t i_op;

	// next variables
	size_t i_var;

	// operation argument indices
	const addr_t*   arg = CPPAD_NULL;

	// initialize the comparision operator counter
	if( p == 0 )
	{	compare_change_number   = 0;
		compare_change_op_index = 0;
	}

	// If this includes a zero calculation, initialize this information
	pod_vector<bool>   isvar_by_ind;
	pod_vector<size_t> index_by_ind;
	if( p == 0 )
	{	size_t i;

		// this includes order zero calculation, initialize vector indices
		size_t num = play->num_vec_ind_rec();
		if( num > 0 )
		{	isvar_by_ind.extend(num);
			index_by_ind.extend(num);
			for(i = 0; i < num; i++)
			{	index_by_ind[i] = play->GetVecInd(i);
				isvar_by_ind[i] = false;
			}
		}
		// includes zero order, so initialize conditional skip flags
		num = play->num_op_rec();
		for(i = 0; i < num; i++)
			cskip_op[i] = false;
	}

	// work space used by UserOp.
	vector<bool> user_vx;        // empty vecotor
	vector<bool> user_vy;        // empty vecotor
	vector<Base> user_tx;        // argument vector Taylor coefficients
	vector<Base> user_ty;        // result vector Taylor coefficients
	//
	atomic_base<Base>* user_atom = CPPAD_NULL; // user's atomic op calculator
# ifndef NDEBUG
	bool               user_ok   = false;      // atomic op return value
# endif
	//
	// information defined by forward_user
	size_t user_old=0, user_m=0, user_n=0, user_i=0, user_j=0;
	enum_user_state user_state = start_user; // proper initialization

	// length of the parameter vector (used by CppAD assert macros)
	const size_t num_par = play->num_par_rec();

	// pointer to the beginning of the parameter vector
	const Base* parameter = CPPAD_NULL;
	if( num_par > 0 )
		parameter = play->GetPar();

	// length of the text vector (used by CppAD assert macros)
	const size_t num_text = play->num_text_rec();

	// pointer to the beginning of the text vector
	const char* text = CPPAD_NULL;
	if( num_text > 0 )
		text = play->GetTxt(0);
	/*
	<!-- end forward0sweep_code_define -->
	*/
	// temporary indices
	size_t i, k;

	// number of orders for this user calculation
	// (not needed for order zero)
	const size_t user_q1 = q+1;

	// variable indices for results vector
	// (done differently for order zero).
	vector<size_t> user_iy;

	// skip the BeginOp at the beginning of the recording
	i_op = 0;
	play->get_op_info(i_op, op, arg, i_var);
	CPPAD_ASSERT_UNKNOWN( op == BeginOp );
	//
# if CPPAD_FORWARD1SWEEP_TRACE
	bool user_trace = false;
	std::cout << std::endl;
# endif
	//
	bool flag; // a temporary flag to use in switch cases
	bool more_operators = true;
	while(more_operators)
	{
		// this op
		play->get_op_info(++i_op, op, arg, i_var);
		CPPAD_ASSERT_UNKNOWN( i_op < play->num_op_rec() );

		// check if we are skipping this operation
		while( cskip_op[i_op] )
		{	switch(op)
			{
				case UserOp:
				{	// get information for this user atomic call
					CPPAD_ASSERT_UNKNOWN( user_state == start_user );
					play->get_user_info(op, arg, user_old, user_m, user_n);
					//
					// skip to the second UserOp
					i_op += user_m + user_n;
					play->get_op_info(++i_op, op, arg, i_var);
					CPPAD_ASSERT_UNKNOWN( op == UserOp );
				}
				break;

				default:
				break;
			}
			play->get_op_info(++i_op, op, arg, i_var);
			CPPAD_ASSERT_UNKNOWN( i_op < play->num_op_rec() );
		}

		// action depends on the operator
		switch( op )
		{
			case AbsOp:
			forward_abs_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case AddvvOp:
			forward_addvv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case AddpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_addpv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case AcosOp:
			// sqrt(1 - x * x), acos(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_acos_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AcoshOp:
			// sqrt(x * x - 1), acosh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_acosh_op(p, q, i_var, arg[0], J, taylor);
			break;
# endif
			// -------------------------------------------------

			case AsinOp:
			// sqrt(1 - x * x), asin(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_asin_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AsinhOp:
			// sqrt(1 + x * x), asinh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_asinh_op(p, q, i_var, arg[0], J, taylor);
			break;
# endif
			// -------------------------------------------------

			case AtanOp:
			// 1 + x * x, atan(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_atan_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AtanhOp:
			// 1 - x * x, atanh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_atanh_op(p, q, i_var, arg[0], J, taylor);
			break;
# endif
			// -------------------------------------------------

			case CExpOp:
			forward_cond_op(
				p, q, i_var, arg, num_par, parameter, J, taylor
			);
			break;
			// ---------------------------------------------------

			case CosOp:
			// sin(x), cos(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_cos_op(p, q, i_var, arg[0], J, taylor);
			break;
			// ---------------------------------------------------

			case CoshOp:
			// sinh(x), cosh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_cosh_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case CSkipOp:
			if( p == 0 )
			{	forward_cskip_op_0(
					i_var, arg, num_par, parameter, J, taylor, cskip_op
				);
			}
			break;
			// -------------------------------------------------

			case CSumOp:
			forward_csum_op(
				p, q, i_var, arg, num_par, parameter, J, taylor
			);
			break;
			// -------------------------------------------------

			case DisOp:
			forward_dis_op(p, q, r, i_var, arg, J, taylor);
			break;
			// -------------------------------------------------

			case DivvvOp:
			forward_divvv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case DivpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_divpv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case DivvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			forward_divvp_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case EndOp:
			CPPAD_ASSERT_NARG_NRES(op, 0, 0);
			more_operators = false;
			break;
			// -------------------------------------------------

			case EqpvOp:
			if( ( p == 0 ) & ( compare_change_count > 0 ) )
			{	forward_eqpv_op_0(
					compare_change_number, arg, parameter, J, taylor
				);
				if( compare_change_count == compare_change_number )
					compare_change_op_index = i_op;
			}
			break;
			// -------------------------------------------------

			case EqvvOp:
			if( ( p == 0 ) & ( compare_change_count > 0 ) )
			{	forward_eqvv_op_0(
					compare_change_number, arg, parameter, J, taylor
				);
				if( compare_change_count == compare_change_number )
					compare_change_op_index = i_op;
			}
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case ErfOp:
			CPPAD_ASSERT_UNKNOWN( CPPAD_USE_CPLUSPLUS_2011 );
			forward_erf_op(p, q, i_var, arg, parameter, J, taylor);
			break;
# endif
			// -------------------------------------------------

			case ExpOp:
			forward_exp_op(p, q, i_var, arg[0], J, taylor);
			break;
			// ---------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case Expm1Op:
			forward_expm1_op(p, q, i_var, arg[0], J, taylor);
			break;
# endif
			// ---------------------------------------------------

			case InvOp:
			CPPAD_ASSERT_NARG_NRES(op, 0, 1);
			break;
			// -------------------------------------------------

			case LdpOp:
			if( p == 0 )
			{	forward_load_p_op_0(
					play,
					i_var,
					arg,
					parameter,
					J,
					taylor,
					isvar_by_ind.data(),
					index_by_ind.data(),
					var_by_load_op.data()
				);
				if( p < q ) forward_load_op(
					play,
					op,
					p+1,
					q,
					r,
					J,
					i_var,
					arg,
					var_by_load_op.data(),
					taylor
				);
			}
			else	forward_load_op(
				play,
				op,
				p,
				q,
				r,
				J,
				i_var,
				arg,
				var_by_load_op.data(),
				taylor
			);
			break;
			// -------------------------------------------------

			case LdvOp:
			if( p == 0 )
			{	forward_load_v_op_0(
					play,
					i_var,
					arg,
					parameter,
					J,
					taylor,
					isvar_by_ind.data(),
					index_by_ind.data(),
					var_by_load_op.data()
				);
				if( p < q ) forward_load_op(
					play,
					op,
					p+1,
					q,
					r,
					J,
					i_var,
					arg,
					var_by_load_op.data(),
					taylor
				);
			}
			else	forward_load_op(
				play,
				op,
				p,
				q,
				r,
				J,
				i_var,
				arg,
				var_by_load_op.data(),
				taylor
			);
			break;
			// -------------------------------------------------

			case LepvOp:
			if( ( p == 0 ) & ( compare_change_count > 0 ) )
			{	forward_lepv_op_0(
					compare_change_number, arg, parameter, J, taylor
				);
				if( compare_change_count == compare_change_number )
					compare_change_op_index = i_op;
			}
			break;

			case LevpOp:
			if( ( p == 0 ) & ( compare_change_count > 0 ) )
			{	forward_levp_op_0(
					compare_change_number, arg, parameter, J, taylor
				);
				if( compare_change_count == compare_change_number )
					compare_change_op_index = i_op;
			}
			break;
			// -------------------------------------------------

			case LevvOp:
			if( ( p == 0 ) & ( compare_change_count > 0 ) )
			{	forward_levv_op_0(
					compare_change_number, arg, parameter, J, taylor
				);
				if( compare_change_count == compare_change_number )
					compare_change_op_index = i_op;
			}
			break;
			// -------------------------------------------------

			case LogOp:
			forward_log_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case Log1pOp:
			forward_log1p_op(p, q, i_var, arg[0], J, taylor);
			break;
# endif
			// -------------------------------------------------

			case LtpvOp:
			if( ( p == 0 ) & ( compare_change_count > 0 ) )
			{	forward_ltpv_op_0(
					compare_change_number, arg, parameter, J, taylor
				);
				if( compare_change_count == compare_change_number )
					compare_change_op_index = i_op;
			}
			break;

			case LtvpOp:
			if( ( p == 0 ) & ( compare_change_count > 0 ) )
			{	forward_ltvp_op_0(
					compare_change_number, arg, parameter, J, taylor
				);
				if( compare_change_count == compare_change_number )
					compare_change_op_index = i_op;
			}
			break;
			// -------------------------------------------------

			case LtvvOp:
			if( ( p == 0 ) & ( compare_change_count > 0 ) )
			{	forward_ltvv_op_0(
					compare_change_number, arg, parameter, J, taylor
				);
				if( compare_change_count == compare_change_number )
					compare_change_op_index = i_op;
			}
			break;
			// -------------------------------------------------

			case MulpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_mulpv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case MulvvOp:
			forward_mulvv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case NepvOp:
			if( ( p == 0 ) & ( compare_change_count > 0 ) )
			{	forward_nepv_op_0(
					compare_change_number, arg, parameter, J, taylor
				);
				if( compare_change_count == compare_change_number )
					compare_change_op_index = i_op;
			}
			break;
			// -------------------------------------------------

			case NevvOp:
			if( ( p == 0 ) & ( compare_change_count > 0 ) )
			{	forward_nevv_op_0(
					compare_change_number, arg, parameter, J, taylor
				);
				if( compare_change_count == compare_change_number )
					compare_change_op_index = i_op;
			}
			break;
			// -------------------------------------------------

			case ParOp:
			i = p;
			if( i == 0 )
			{	forward_par_op_0(
					i_var, arg, num_par, parameter, J, taylor
				);
				i++;
			}
			while(i <= q)
			{	taylor[ i_var * J + i] = Base(0.0);
				i++;
			}
			break;
			// -------------------------------------------------

			case PowvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			forward_powvp_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case PowpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_powpv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case PowvvOp:
			forward_powvv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case PriOp:
			if( (p == 0) & print ) forward_pri_0(s_out,
				arg, num_text, text, num_par, parameter, J, taylor
			);
			break;
			// -------------------------------------------------

			case SignOp:
			// sign(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_sign_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case SinOp:
			// cos(x), sin(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_sin_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case SinhOp:
			// cosh(x), sinh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_sinh_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case SqrtOp:
			forward_sqrt_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case StppOp:
			if( p == 0 )
			{	forward_store_pp_op_0(
					i_var,
					arg,
					num_par,
					J,
					taylor,
					isvar_by_ind.data(),
					index_by_ind.data()
				);
			}
			break;
			// -------------------------------------------------

			case StpvOp:
			if( p == 0 )
			{	forward_store_pv_op_0(
					i_var,
					arg,
					num_par,
					J,
					taylor,
					isvar_by_ind.data(),
					index_by_ind.data()
				);
			}
			break;
			// -------------------------------------------------

			case StvpOp:
			if( p == 0 )
			{	forward_store_vp_op_0(
					i_var,
					arg,
					num_par,
					J,
					taylor,
					isvar_by_ind.data(),
					index_by_ind.data()
				);
			}
			break;
			// -------------------------------------------------

			case StvvOp:
			if( p == 0 )
			{	forward_store_vv_op_0(
					i_var,
					arg,
					num_par,
					J,
					taylor,
					isvar_by_ind.data(),
					index_by_ind.data()
				);
			}
			break;
			// -------------------------------------------------

			case SubvvOp:
			forward_subvv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case SubpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_subpv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case SubvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			forward_subvp_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case TanOp:
			// tan(x)^2, tan(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_tan_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case TanhOp:
			// tanh(x)^2, tanh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_tanh_op(p, q, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case UserOp:
			// start or end an atomic function call
			flag = user_state == start_user;
			user_atom = play->get_user_info(op, arg, user_old, user_m, user_n);
			if( flag )
			{	user_state = arg_user;
				user_i     = 0;
				user_j     = 0;
				//
				user_tx.resize(user_n * user_q1);
				user_ty.resize(user_m * user_q1);
				user_iy.resize(user_m);
			}
			else
			{	user_state = start_user;
				//
				// call users function for this operation
				user_atom->set_old(user_old);
				CPPAD_ATOMIC_CALL(
					p, q, user_vx, user_vy, user_tx, user_ty
				);
# ifndef NDEBUG
				if( ! user_ok )
				{	std::string msg =
						user_atom->afun_name()
						+ ": atomic_base.forward: returned false";
					CPPAD_ASSERT_KNOWN(false, msg.c_str() );
				}
# endif
				for(i = 0; i < user_m; i++)
					if( user_iy[i] > 0 )
						for(k = p; k <= q; k++)
							taylor[ user_iy[i] * J + k ] =
								user_ty[ i * user_q1 + k ];
# if CPPAD_FORWARD1SWEEP_TRACE
				user_trace = true;
# endif
			}
			break;

			case UsrapOp:
			// parameter argument for a user atomic function
			CPPAD_ASSERT_UNKNOWN( NumArg(op) == 1 );
			CPPAD_ASSERT_UNKNOWN( user_state == arg_user );
			CPPAD_ASSERT_UNKNOWN( user_i == 0 );
			CPPAD_ASSERT_UNKNOWN( user_j < user_n );
			CPPAD_ASSERT_UNKNOWN( size_t( arg[0] ) < num_par );
			//
			user_tx[user_j * user_q1 + 0] = parameter[ arg[0]];
			for(k = 1; k < user_q1; k++)
				user_tx[user_j * user_q1 + k] = Base(0.0);
			//
			++user_j;
			if( user_j == user_n )
				user_state = ret_user;
			break;

			case UsravOp:
			// variable argument for a user atomic function
			CPPAD_ASSERT_UNKNOWN( NumArg(op) == 1 );
			CPPAD_ASSERT_UNKNOWN( user_state == arg_user );
			CPPAD_ASSERT_UNKNOWN( user_i == 0 );
			CPPAD_ASSERT_UNKNOWN( user_j < user_n );
			//
			for(k = 0; k < user_q1; k++)
				user_tx[user_j * user_q1 + k] = taylor[ arg[0] * J + k];
			//
			++user_j;
			if( user_j == user_n )
				user_state = ret_user;
			break;

			case UsrrpOp:
			// parameter result for a user atomic function
			CPPAD_ASSERT_NARG_NRES(op, 1, 0);
			CPPAD_ASSERT_UNKNOWN( user_state == ret_user );
			CPPAD_ASSERT_UNKNOWN( user_i < user_m );
			CPPAD_ASSERT_UNKNOWN( user_j == user_n );
			CPPAD_ASSERT_UNKNOWN( size_t( arg[0] ) < num_par );
			//
			user_iy[user_i] = 0;
			user_ty[user_i * user_q1 + 0] = parameter[ arg[0]];
			for(k = 1; k < p; k++)
				user_ty[user_i * user_q1 + k] = Base(0.0);
			//
			++user_i;
			if( user_i == user_m )
				user_state = end_user;
			break;

			case UsrrvOp:
			// variable result for a user atomic function
			CPPAD_ASSERT_NARG_NRES(op, 0, 1);
			CPPAD_ASSERT_UNKNOWN( user_state == ret_user );
			CPPAD_ASSERT_UNKNOWN( user_i < user_m );
			CPPAD_ASSERT_UNKNOWN( user_j == user_n );
			//
			user_iy[user_i] = i_var;
			for(k = 0; k < p; k++)
				user_ty[user_i * user_q1 + k] = taylor[ i_var * J + k];
			//
			++user_i;
			if( user_i == user_m )
				user_state = end_user;
			break;
			// -------------------------------------------------

			case ZmulpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_zmulpv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case ZmulvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			forward_zmulvp_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case ZmulvvOp:
			forward_zmulvv_op(p, q, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			default:
			CPPAD_ASSERT_UNKNOWN(0);
		}
# if CPPAD_FORWARD1SWEEP_TRACE
		if( user_trace )
		{	user_trace = false;

			CPPAD_ASSERT_UNKNOWN( op == UserOp );
			CPPAD_ASSERT_UNKNOWN( NumArg(UsrrvOp) == 0 );
			for(i = 0; i < user_m; i++) if( user_iy[i] > 0 )
			{	size_t i_tmp   = (i_op + i) - user_m;
				printOp(
					std::cout,
					play,
					i_tmp,
					user_iy[i],
					UsrrvOp,
					CPPAD_NULL
				);
				Base* Z_tmp = taylor + user_iy[i] * J;
				printOpResult(
					std::cout,
					q + 1,
					Z_tmp,
					0,
					(Base *) CPPAD_NULL
				);
				std::cout << std::endl;
			}
		}
		Base*           Z_tmp   = taylor + J * i_var;
		if( op != UsrrvOp )
		{
			printOp(
				std::cout,
				play,
				i_op,
				i_var,
				op,
				arg
			);
			if( NumRes(op) > 0 ) printOpResult(
				std::cout,
				q + 1,
				Z_tmp,
				0,
				(Base *) CPPAD_NULL
			);
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
# else
	}
# endif
	CPPAD_ASSERT_UNKNOWN( user_state == start_user );

	if( (p == 0) & (compare_change_count == 0) )
		compare_change_number = 0;
	return;
}

// preprocessor symbols that are local to this file
# undef CPPAD_FORWARD1SWEEP_TRACE
# undef CPPAD_ATOMIC_CALL

} } // END_CPPAD_LOCAL_NAMESPACE
# endif

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2021 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not visit scipopt.org.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file    nlpi_ipopt.cpp
 * @ingroup NLPIS
 * @brief   Ipopt NLP interface
 * @author  Stefan Vigerske
 * @author  Benjamin Müller
 *
 * @todo if too few degrees of freedom, solve a slack-minimization problem instead?
 *
 * This file can only be compiled if Ipopt is available.
 * Otherwise, to resolve public functions, use nlpi_ipopt_dummy.c.
 * Since the dummy code is C instead of C++, it has been moved into a separate file.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "scip/nlpi_ipopt.h"

#include "scip/nlpioracle.h"
#include "scip/exprinterpret.h"
#include "scip/scip_nlpi.h"
#include "scip/scip_nlp.h"
#include "scip/scip_randnumgen.h"
#include "scip/scip_mem.h"
#include "scip/scip_message.h"
#include "scip/scip_general.h"
#include "scip/scip_numerics.h"
#include "scip/scip_param.h"
#include "scip/scip_solve.h"
#include "scip/pub_misc.h"
#include "scip/pub_paramset.h"

#include <new>      /* for std::bad_alloc */
#include <sstream>
#include <cstring>

/* turn off some lint warnings for file */
/*lint --e{1540,750,3701}*/

#include "IpoptConfig.h"

#if defined(__GNUC__) && IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR < 14
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "IpIpoptApplication.hpp"
#include "IpIpoptCalculatedQuantities.hpp"
#include "IpSolveStatistics.hpp"
#include "IpJournalist.hpp"
#include "IpIpoptData.hpp"
#include "IpTNLPAdapter.hpp"
#include "IpOrigIpoptNLP.hpp"
#include "IpLapack.hpp"
#if defined(__GNUC__) && IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR < 14
#pragma GCC diagnostic warning "-Wshadow"
#endif

#if (IPOPT_VERSION_MAJOR < 3 || (IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR < 12) || (IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR == 12 && IPOPT_VERSION_RELEASE < 5))
#error "The Ipopt interface requires at least 3.12.5"
#endif

/* MUMPS that can be used by Ipopt is not threadsafe
 * If we want SCIP to be threadsafe (SCIP_THREADSAFE), have std::mutex (C++11 or higher), and use Ipopt before 3.14,
 * then we protect the call to Ipopt by a mutex if MUMPS is used as linear solver.
 * Thus, we allow only one Ipopt run at a time.
 * Ipopt 3.14 has this build-in to its MUMPS interface, so we won't have to take care of this.
 */
#if defined(SCIP_THREADSAFE) && __cplusplus >= 201103L && IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR < 14
#define PROTECT_SOLVE_BY_MUTEX
#include <mutex>
static std::mutex solve_mutex;  /*lint !e1756*/
#endif

using namespace Ipopt;

#define NLPI_NAME          "ipopt"           /**< short concise name of solver */
#define NLPI_DESC          "Ipopt interface" /**< description of solver */
#define NLPI_PRIORITY      1000              /**< priority */

#define MAXPERTURB         0.01              /**< maximal perturbation of bounds in starting point heuristic */
#define FEASTOLFACTOR      0.9               /**< factor for user-given feasibility tolerance to get feasibility tolerance that is actually passed to Ipopt */

#define DEFAULT_RANDSEED   71                /**< initial random seed */


/* Convergence check (see ScipNLP::intermediate_callback)
 *
 * If the fastfail option is set to 2, then we stop Ipopt if the reduction in
 * primal infeasibility is not sufficient for a consecutive number of iterations.
 * With the parameters as given below, we require Ipopt to
 * - not increase the primal infeasibility after 5 iterations
 * - reduce the primal infeasibility by at least 50% within 10 iterations
 * - reduce the primal infeasibility by at least 90% within 30 iterations
 * The targets are updated once they are reached and the limit on allowed iterations to reach the new target is reset.
 *
 * In certain situations, it is allowed to exceed an iteration limit:
 * - If we are in the first 10 (convcheck_startiter) iterations.
 * - If we are within 10 (convcheck_startiter) iterations after the restoration phase ended.
 *   The reason for this is that during feasibility restoration phase Ipopt aims completely on
 *   reducing constraint violation, completely forgetting the objective function.
 *   When returning from feasibility restoration and considering the original objective again,
 *   it is unlikely that Ipopt will continue to decrease primal infeasibility, since it may now target on
 *   more on optimality again. Thus, we do not check convergence for a number of iterations.
 * - If the target on dual infeasibility reduction has been achieved, we are below twice the iteration limit, and
 *   we are not in restoration mode.
 *   The reason for this is that if Ipopt makes good progress towards optimality,
 *   we want to allow some more iterations where primal infeasibility is not reduced.
 *   However, in restoration mode, dual infeasibility does not correspond to the original problem and
 *   the complete aim is to restore primal infeasibility.
 */
static const int convcheck_nchecks                         = 3;                 /**< number of convergence checks */
static const int convcheck_startiter                       = 10;                /**< iteration where to start convergence checking */
static const int convcheck_maxiter[convcheck_nchecks]      = { 5,   15,  30 };  /**< maximal number of iterations to achieve each convergence check */
static const SCIP_Real convcheck_minred[convcheck_nchecks] = { 1.0, 0.5, 0.1 }; /**< minimal required infeasibility reduction in each convergence check */

/// integer parameters of Ipopt to make available via SCIP parameters
static const char* ipopt_int_params[] =
   { "print_level" };   // print_level must be first

/// string parameters of Ipopt to make available via SCIP parameters
static const char* ipopt_string_params[] =
   { "linear_solver",
     "hsllib",
     "pardisolib",
     "linear_system_scaling",
     "nlp_scaling_method",
     "mu_strategy",
     "hessian_approximation"
   };

class ScipNLP;

struct SCIP_NlpiData
{
public:
   char*                       optfile;      /**< Ipopt options file to read */
   std::string                 defoptions;   /**< modified default options for Ipopt */

   /** constructor */
   explicit SCIP_NlpiData()
   : optfile(NULL)
   { }
};

struct SCIP_NlpiProblem
{
public:
   SCIP_NLPIORACLE*            oracle;       /**< Oracle-helper to store and evaluate NLP */
   SCIP_RANDNUMGEN*            randnumgen;   /**< random number generator */

   SmartPtr<IpoptApplication>  ipopt;        /**< Ipopt application */
   SmartPtr<ScipNLP>           nlp;          /**< NLP in Ipopt form */

   bool                        printlevelset;/**< whether Ipopt print_level was set via nlpi/ipopt/print_level option */
   bool                        firstrun;     /**< whether the next NLP solve will be the first one */
   bool                        samestructure;/**< whether the NLP solved next will still have the same (Ipopt-internal) structure (same number of variables, constraints, bounds, and nonzero pattern) */

   SCIP_NLPSOLSTAT             solstat;      /**< status of current solution (if any) */
   SCIP_NLPTERMSTAT            termstat;     /**< termination status of last solve (if any) */
   bool                        solprimalvalid;/**< whether primal solution values are available (solprimals has meaningful values) */
   bool                        solprimalgiven;/**< whether primal solution values were set by caller */
   bool                        soldualvalid; /**< whether dual solution values are available (soldual* have meaningful values) */
   bool                        soldualgiven; /**< whether dual solution values were set by caller */
   SCIP_Real*                  solprimals;   /**< primal solution values, if available */
   SCIP_Real*                  soldualcons;  /**< dual solution values of constraints, if available */
   SCIP_Real*                  soldualvarlb; /**< dual solution values of variable lower bounds, if available */
   SCIP_Real*                  soldualvarub; /**< dual solution values of variable upper bounds, if available */
   SCIP_Real                   solobjval;    /**< objective function value in solution from last run */
   int                         lastniter;    /**< number of iterations in last run */
   SCIP_Real                   lasttime;     /**< time spend in last run */

   /** constructor */
   SCIP_NlpiProblem()
      : oracle(NULL), randnumgen(NULL),
        printlevelset(false), firstrun(true), samestructure(true),
        solstat(SCIP_NLPSOLSTAT_UNKNOWN), termstat(SCIP_NLPTERMSTAT_OTHER),
        solprimalvalid(false), solprimalgiven(false), soldualvalid(false), soldualgiven(false),
        solprimals(NULL), soldualcons(NULL), soldualvarlb(NULL), soldualvarub(NULL),
        solobjval(SCIP_INVALID), lastniter(-1), lasttime(-1.0)
   { }
};

/** TNLP implementation for SCIPs NLP */
class ScipNLP : public TNLP
{
private:
   SCIP_NLPIPROBLEM*     nlpiproblem;        /**< NLPI problem data */
   SCIP*                 scip;               /**< SCIP data structure */
   SCIP_NLPPARAM         param;              /**< NLP solve parameters */

   SCIP_Real             conv_prtarget[convcheck_nchecks]; /**< target primal infeasibility for each convergence check */
   SCIP_Real             conv_dutarget[convcheck_nchecks]; /**< target dual infeasibility for each convergence check */
   int                   conv_iterlim[convcheck_nchecks];  /**< iteration number where target primal infeasibility should to be achieved */
   int                   conv_lastrestoiter;               /**< last iteration number in restoration mode, or -1 if none */

   unsigned int          current_x;          /**< unique number that identifies current iterate (x): incremented when Ipopt calls with new_x=true */
   unsigned int          last_f_eval_x;      /**< the number of the iterate for which the objective was last evaluated (eval_f) */
   unsigned int          last_g_eval_x;      /**< the number of the iterate for which the constraints were last evaluated (eval_g) */

public:
   bool                  approxhessian;      /**< do we tell Ipopt to approximate the hessian? (may also be false if user set to approx. hessian via option file) */

   // cppcheck-suppress uninitMemberVar
   /** constructor */
   ScipNLP(
      SCIP_NLPIPROBLEM*  nlpiproblem_ = NULL,/**< NLPI problem data */
      SCIP*              scip_ = NULL        /**< SCIP data structure */
      )
      : nlpiproblem(nlpiproblem_), scip(scip_),
        conv_lastrestoiter(-1),
        current_x(1), last_f_eval_x(0), last_g_eval_x(0),
        approxhessian(false)
   {
      assert(scip != NULL);
   }

   /** destructor */
   ~ScipNLP()
   { /*lint --e{1540}*/
   }

   /** initialize for new solve */
   void initializeSolve(
      SCIP_NLPIPROBLEM*    nlpiproblem_,     /**< NLPI problem */
      const SCIP_NLPPARAM& nlpparam          /**< NLP solve parameters */
      )
   {
      assert(nlpiproblem_ != NULL);
      nlpiproblem = nlpiproblem_;
      param = nlpparam;

      // it appears we are about to start a new solve
      // use this call as an opportunity to reset the counts on x
      current_x = 1;
      last_f_eval_x = 0;
      last_g_eval_x = 0;
   }

   /** Method to return some info about the nlp */
   bool get_nlp_info(
      Index&             n,                  /**< place to store number of variables */ 
      Index&             m,                  /**< place to store number of constraints */ 
      Index&             nnz_jac_g,          /**< place to store number of nonzeros in jacobian */
      Index&             nnz_h_lag,          /**< place to store number of nonzeros in hessian */
      IndexStyleEnum&    index_style         /**< place to store used index style (0-based or 1-based) */
      );

   /** Method to return the bounds for my problem */
   bool get_bounds_info(
      Index              n,                  /**< number of variables */ 
      Number*            x_l,                /**< buffer to store lower bounds on variables */
      Number*            x_u,                /**< buffer to store upper bounds on variables */
      Index              m,                  /**< number of constraints */
      Number*            g_l,                /**< buffer to store lower bounds on constraints */
      Number*            g_u                 /**< buffer to store lower bounds on constraints */
      );

   /** Method to return the starting point for the algorithm */
   bool get_starting_point(
      Index              n,                  /**< number of variables */ 
      bool               init_x,             /**< whether initial values for primal values are requested */ 
      Number*            x,                  /**< buffer to store initial primal values */
      bool               init_z,             /**< whether initial values for dual values of variable bounds are requested */  
      Number*            z_L,                /**< buffer to store dual values for variable lower bounds */
      Number*            z_U,                /**< buffer to store dual values for variable upper bounds */
      Index              m,                  /**< number of constraints */
      bool               init_lambda,        /**< whether initial values for dual values of constraints are required */
      Number*            lambda              /**< buffer to store dual values of constraints */
      );

   /** Method to return the number of nonlinear variables. */
   Index get_number_of_nonlinear_variables();

   /** Method to return the indices of the nonlinear variables */
   bool get_list_of_nonlinear_variables(
      Index              num_nonlin_vars,    /**< number of nonlinear variables */
      Index*             pos_nonlin_vars     /**< array to fill with indices of nonlinear variables */
      );

   /** Method to return metadata about variables and constraints */
   bool get_var_con_metadata(
      Index              n,                  /**< number of variables */
      StringMetaDataMapType& var_string_md,  /**< variable meta data of string type */
      IntegerMetaDataMapType& var_integer_md,/**< variable meta data of integer type */
      NumericMetaDataMapType& var_numeric_md,/**< variable meta data of numeric type */
      Index              m,                  /**< number of constraints */
      StringMetaDataMapType& con_string_md,  /**< constraint meta data of string type */
      IntegerMetaDataMapType& con_integer_md,/**< constraint meta data of integer type */
      NumericMetaDataMapType& con_numeric_md /**< constraint meta data of numeric type */
      );

   /** Method to return the objective value */
   bool eval_f(
      Index              n,                  /**< number of variables */ 
      const Number*      x,                  /**< point to evaluate */ 
      bool               new_x,              /**< whether some function evaluation method has been called for this point before */
      Number&            obj_value           /**< place to store objective function value */
      );

   /** Method to return the gradient of the objective */
   bool eval_grad_f(
      Index              n,                  /**< number of variables */ 
      const Number*      x,                  /**< point to evaluate */ 
      bool               new_x,              /**< whether some function evaluation method has been called for this point before */
      Number*            grad_f              /**< buffer to store objective gradient */
      );

   /** Method to return the constraint residuals */
   bool eval_g(
      Index              n,                  /**< number of variables */ 
      const Number*      x,                  /**< point to evaluate */ 
      bool               new_x,              /**< whether some function evaluation method has been called for this point before */
      Index              m,                  /**< number of constraints */
      Number*            g                   /**< buffer to store constraint function values */
      );

   /** Method to return:
    *   1) The structure of the jacobian (if "values" is NULL)
    *   2) The values of the jacobian (if "values" is not NULL)
    */
   bool eval_jac_g(
      Index              n,                  /**< number of variables */ 
      const Number*      x,                  /**< point to evaluate */ 
      bool               new_x,              /**< whether some function evaluation method has been called for this point before */
      Index              m,                  /**< number of constraints */
      Index              nele_jac,           /**< number of nonzero entries in jacobian */ 
      Index*             iRow,               /**< buffer to store row indices of nonzero jacobian entries, or NULL if values 
                                              * are requested */
      Index*             jCol,               /**< buffer to store column indices of nonzero jacobian entries, or NULL if values
                                              * are requested */
      Number*            values              /**< buffer to store values of nonzero jacobian entries, or NULL if structure is
                                              * requested */
      );

   /** Method to return:
    *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
    *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
    */
   bool eval_h(
      Index              n,                  /**< number of variables */ 
      const Number*      x,                  /**< point to evaluate */ 
      bool               new_x,              /**< whether some function evaluation method has been called for this point before */
      Number             obj_factor,         /**< weight for objective function */ 
      Index              m,                  /**< number of constraints */
      const Number*      lambda,             /**< weights for constraint functions */ 
      bool               new_lambda,         /**< whether the hessian has been evaluated for these values of lambda before */
      Index              nele_hess,          /**< number of nonzero entries in hessian */
      Index*             iRow,               /**< buffer to store row indices of nonzero hessian entries, or NULL if values
                                              * are requested */
      Index*             jCol,               /**< buffer to store column indices of nonzero hessian entries, or NULL if values
                                              * are requested */
      Number*            values              /**< buffer to store values of nonzero hessian entries, or NULL if structure is requested */
      );

   /** Method called by the solver at each iteration.
    * 
    * Checks whether Ctrl-C was hit.
    */
   bool intermediate_callback(
      AlgorithmMode      mode,               /**< current mode of algorithm */
      Index              iter,               /**< current iteration number */
      Number             obj_value,          /**< current objective value */
      Number             inf_pr,             /**< current primal infeasibility */
      Number             inf_du,             /**< current dual infeasibility */
      Number             mu,                 /**< current barrier parameter */
      Number             d_norm,             /**< current gradient norm */
      Number             regularization_size,/**< current size of regularization */
      Number             alpha_du,           /**< current dual alpha */
      Number             alpha_pr,           /**< current primal alpha */
      Index              ls_trials,          /**< current number of linesearch trials */
      const IpoptData*   ip_data,            /**< pointer to Ipopt Data */
      IpoptCalculatedQuantities* ip_cq       /**< pointer to current calculated quantities */
      );

   /** This method is called when the algorithm is complete so the TNLP can store/write the solution. */
   void finalize_solution(
      SolverReturn       status,             /**< solve and solution status */ 
      Index              n,                  /**< number of variables */ 
      const Number*      x,                  /**< primal solution values */ 
      const Number*      z_L,                /**< dual values of variable lower bounds */
      const Number*      z_U,                /**< dual values of variable upper bounds */
      Index              m,                  /**< number of constraints */ 
      const Number*      g,                  /**< values of constraints */ 
      const Number*      lambda,             /**< dual values of constraints */ 
      Number             obj_value,          /**< objective function value */ 
      const IpoptData*   data,               /**< pointer to Ipopt Data */ 
      IpoptCalculatedQuantities* cq          /**< pointer to calculated quantities */
      );
};

/** A particular Ipopt::Journal implementation that uses the SCIP message routines for output. */
class ScipJournal : public Ipopt::Journal {
private:
   SCIP*                 scip;               /**< SCIP data structure */

public:
   ScipJournal(
      const char*          name,             /**< name of journal */
      Ipopt::EJournalLevel default_level,    /**< default verbosity level */
      SCIP*                scip_             /**< SCIP data structure */
      )
      : Ipopt::Journal(name, default_level),
        scip(scip_)
   { }

   ~ScipJournal() { }

protected:
   /*lint -e{715}*/
   void PrintImpl(
      Ipopt::EJournalCategory category,      /**< category of message */
      Ipopt::EJournalLevel    level,         /**< verbosity level of message */
      const char*             str            /**< message to print */
      )
   {  /*lint --e{715} */
      if( level == J_ERROR )
      {
         SCIPerrorMessage("%s", str);
      }
      else
      {
         SCIP_VERBLEVEL msgverblevel;
         if( level <= J_WARNING )
            msgverblevel = SCIP_VERBLEVEL_DIALOG;
         else if( level <= J_SUMMARY )
            msgverblevel = SCIP_VERBLEVEL_MINIMAL;
         else
            msgverblevel = SCIP_VERBLEVEL_HIGH;
         SCIPverbMessage(scip, msgverblevel, NULL, "%s", str);
      }
   }

   /*lint -e{715}*/
   void PrintfImpl(
      Ipopt::EJournalCategory category,      /**< category of message */
      Ipopt::EJournalLevel    level,         /**< verbosity level of message */
      const char*             pformat,       /**< message printing format */
      va_list                 ap             /**< arguments of message */
      )
   {  /*lint --e{715} */
      if( level == J_ERROR )
      {
         SCIPmessagePrintErrorHeader(__FILE__, __LINE__);
         SCIPmessageVPrintError(pformat, ap);
      }
      else
      {
         SCIP_VERBLEVEL msgverblevel;
         if( level <= J_WARNING )
            msgverblevel = SCIP_VERBLEVEL_DIALOG;
         else if( level <= J_SUMMARY )
            msgverblevel = SCIP_VERBLEVEL_MINIMAL;
         else
            msgverblevel = SCIP_VERBLEVEL_HIGH;
         SCIPmessageVPrintVerbInfo(SCIPgetMessagehdlr(scip), SCIPgetVerbLevel(scip), msgverblevel, pformat, ap);
      }
   }

   void FlushBufferImpl() { }
};

/** sets status codes to mark that last NLP solve is no longer valid (usually because the NLP changed) */
static
void invalidateSolved(
   SCIP_NLPIPROBLEM*     problem             /**< data structure of problem */
   )
{
   problem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
   problem->termstat = SCIP_NLPTERMSTAT_OTHER;
   problem->solobjval = SCIP_INVALID;
   problem->lastniter = -1;
   problem->lasttime  = -1.0;
}

/** sets solution values to be invalid and calls invalidateSolved() */
static
void invalidateSolution(
   SCIP_NLPIPROBLEM*     problem             /**< data structure of problem */
   )
{
   assert(problem != NULL);

   problem->solprimalvalid = false;
   problem->solprimalgiven = false;
   problem->soldualvalid = false;
   problem->soldualgiven = false;

   invalidateSolved(problem);
}

/** makes sure a starting point (initial guess) is available */
static
SCIP_RETCODE ensureStartingPoint(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPIPROBLEM*     problem,            /**< data structure of problem */
   SCIP_Bool&            warmstart           /**< whether a warmstart has been request */
   )
{
   SCIP_Real lb, ub;
   int n;

   assert(problem != NULL);

   // disable warmstart if no primal or dual solution values are available
   if( warmstart && (!problem->solprimalvalid || !problem->soldualvalid ))
   {
      SCIPdebugMsg(scip, "Disable warmstart as no primal or dual solution available.\n");
      warmstart = false;
   }

   // continue below with making up a random primal starting point if
   // the user did not set a starting point and warmstart is disabled (so the last solution shouldn't be used)
   // (if warmstart, then due to the checks above we must now have valid primal and dual solution values)
   if( problem->solprimalgiven || warmstart )
   {
      // so we must have a primal solution to start from
      // if warmstart, then we also need to have a dual solution to start from
      assert(problem->solprimalvalid);
      assert(problem->solprimals != NULL);
      assert(!warmstart || problem->soldualgiven);
      assert(!warmstart || problem->soldualcons != NULL);
      assert(!warmstart || problem->soldualvarlb != NULL);
      assert(!warmstart || problem->soldualvarub != NULL);
      SCIPdebugMsg(scip, "Starting solution for %sstart available from %s.\n",
         warmstart ? "warm" : "cold",
         problem->solprimalgiven ? "user" : "previous solve");
      return SCIP_OKAY;
   }

   SCIPdebugMsg(scip, "Starting solution for coldstart not available. Making up something by projecting 0 onto variable bounds and adding a random perturbation.\n");

   n = SCIPnlpiOracleGetNVars(problem->oracle);

   if( problem->randnumgen == NULL )
   {
      SCIP_CALL( SCIPcreateRandom(scip, &problem->randnumgen, DEFAULT_RANDSEED, TRUE) );
   }

   if( problem->solprimals == NULL )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &problem->solprimals, n) );
   }

   for( int i = 0; i < n; ++i )
   {
      lb = SCIPnlpiOracleGetVarLbs(problem->oracle)[i];
      ub = SCIPnlpiOracleGetVarUbs(problem->oracle)[i];
      if( lb > 0.0 )
         problem->solprimals[i] = SCIPrandomGetReal(problem->randnumgen, lb, lb + MAXPERTURB*MIN(1.0, ub-lb));
      else if( ub < 0.0 )
         problem->solprimals[i] = SCIPrandomGetReal(problem->randnumgen, ub - MAXPERTURB*MIN(1.0, ub-lb), ub);
      else
         problem->solprimals[i] = SCIPrandomGetReal(problem->randnumgen, MAX(lb, -MAXPERTURB*MIN(1.0, ub-lb)), MIN(ub, MAXPERTURB*MIN(1.0, ub-lb)));
   }
   problem->solprimalvalid = true;

   return SCIP_OKAY;
}

/// pass NLP solve parameters to Ipopt
static
SCIP_RETCODE handleNlpParam(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPIPROBLEM*     nlpiproblem,        /**< NLP */
   const SCIP_NLPPARAM   param               /**< solve parameters */
   )
{
   assert(scip != NULL);
   assert(nlpiproblem != NULL);

   if( !nlpiproblem->printlevelset )
   {
      switch( param.verblevel )
      {
         case 0:
            (void) nlpiproblem->ipopt->Options()->SetIntegerValue("print_level", J_ERROR);
            break;
         case 1:
            (void) nlpiproblem->ipopt->Options()->SetIntegerValue("print_level", J_ITERSUMMARY);
            break;
         case 2:
            (void) nlpiproblem->ipopt->Options()->SetIntegerValue("print_level", J_DETAILED);
            break;
         default:
            (void) nlpiproblem->ipopt->Options()->SetIntegerValue("print_level", MIN(J_ITERSUMMARY + (param.verblevel-1), J_ALL));
            break;
      }
   }

   (void) nlpiproblem->ipopt->Options()->SetIntegerValue("max_iter", param.iterlimit);

   (void) nlpiproblem->ipopt->Options()->SetNumericValue("constr_viol_tol", FEASTOLFACTOR * param.feastol);
   (void) nlpiproblem->ipopt->Options()->SetNumericValue("acceptable_constr_viol_tol", FEASTOLFACTOR * param.feastol);

   /* set optimality tolerance parameters in Ipopt
    *
    * Sets dual_inf_tol, compl_inf_tol, and tol to relobjtol.
    * We leave acceptable_dual_inf_tol and acceptable_compl_inf_tol untouched for now, which means that if Ipopt has convergence problems, then
    * it can stop with a solution that is still feasible, but essentially without a proof of local optimality.
    * Note, that in this case we report only feasibility and not optimality of the solution (see ScipNLP::finalize_solution).
    *
    * TODO it makes sense to set tol (maximal errors in scaled problem) depending on user parameters somewhere
    *      NLPI parameter RELOBJTOL seems better suited for this than FEASTOL, but maybe we need another one?
    */
   (void) nlpiproblem->ipopt->Options()->SetNumericValue("dual_inf_tol", param.relobjtol);
   (void) nlpiproblem->ipopt->Options()->SetNumericValue("compl_inf_tol", param.relobjtol);
   (void) nlpiproblem->ipopt->Options()->SetNumericValue("tol", param.relobjtol);

   /* Ipopt doesn't like a setting of exactly 0 for the max_*_time, so increase as little as possible in that case */
#if IPOPT_VERSION_MAJOR > 3 || IPOPT_VERSION_MINOR >= 14
   (void) nlpiproblem->ipopt->Options()->SetNumericValue("max_wall_time", MAX(param.timelimit, DBL_MIN));
#else
   (void) nlpiproblem->ipopt->Options()->SetNumericValue("max_cpu_time", MAX(param.timelimit, DBL_MIN));
#endif

   // disable acceptable-point heuristic iff fastfail is completely off
   // it seems useful to have Ipopt stop when it obviously doesn't make progress (like one of the NLPs in the bendersqp ctest)
   if( param.fastfail == 0 )
      (void) nlpiproblem->ipopt->Options()->SetIntegerValue("acceptable_iter", 0);
   else
#if IPOPT_VERSION_MAJOR > 3 || IPOPT_VERSION_MINOR > 14 || (IPOPT_VERSION_MINOR == 14 && IPOPT_VERSION_RELEASE >= 2)
      (void) nlpiproblem->ipopt->Options()->UnsetValue("acceptable_iter");
#else
      (void) nlpiproblem->ipopt->Options()->SetIntegerValue("acceptable_iter", 15);  // 15 is the default
#endif

   if( !nlpiproblem->ipopt->Options()->SetStringValue("warm_start_init_point", param.warmstart ? "yes" : "no") && !param.warmstart )
   {
      // if we cannot disable warmstarts in Ipopt, then we have a big problem
      SCIPerrorMessage("Failed to set Ipopt warm_start_init_point option to no.");
      return SCIP_ERROR;
   }

   return SCIP_OKAY;
}

/** copy method of NLP interface (called when SCIP copies plugins)
 *
 *  input:
 *  - blkmem block memory of target SCIP
 *  - sourcenlpi the NLP interface to copy
 *  - targetnlpi buffer to store pointer to copy of NLP interface
 */
static
SCIP_DECL_NLPICOPY(nlpiCopyIpopt)
{
   SCIP_NLPI* targetnlpi;
   SCIP_NLPIDATA* sourcedata;
   SCIP_NLPIDATA* targetdata;

   assert(sourcenlpi != NULL);

   SCIP_CALL( SCIPincludeNlpSolverIpopt(scip) );

   targetnlpi = SCIPfindNlpi(scip, NLPI_NAME);
   assert(targetnlpi != NULL);

   sourcedata = SCIPnlpiGetData(sourcenlpi);
   assert(sourcedata != NULL);

   targetdata = SCIPnlpiGetData(targetnlpi);
   assert(targetdata != NULL);

   targetdata->defoptions = sourcedata->defoptions;

   return SCIP_OKAY;
}

/** destructor of NLP interface to free nlpi data
 * 
 * input:
 *  - nlpi datastructure for solver interface
 */
static
SCIP_DECL_NLPIFREE(nlpiFreeIpopt)
{
   assert(nlpidata != NULL);

   delete *nlpidata;
   *nlpidata = NULL;

   return SCIP_OKAY;
}

/** gets pointer for NLP solver to do dirty stuff
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *
 *  return: void pointer to solver
 */
static
SCIP_DECL_NLPIGETSOLVERPOINTER(nlpiGetSolverPointerIpopt)
{
   assert(nlpi != NULL);

   return NULL;
}

/** creates a problem instance
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem pointer to store the problem data
 *  - name name of problem, can be NULL
 */
static
SCIP_DECL_NLPICREATEPROBLEM(nlpiCreateProblemIpopt)
{
   SCIP_NLPIDATA* data;

   assert(nlpi    != NULL);
   assert(problem != NULL);

   data = SCIPnlpiGetData(nlpi);
   assert(data != NULL);

   SCIP_ALLOC( *problem = new SCIP_NLPIPROBLEM ); /*lint !e774*/

   SCIP_CALL( SCIPnlpiOracleCreate(scip, &(*problem)->oracle) );
   SCIP_CALL( SCIPnlpiOracleSetProblemName(scip, (*problem)->oracle, name) );

   try
   {
      /* initialize IPOPT without default journal */
      (*problem)->ipopt = new IpoptApplication(false);

      /* plugin our journal to get output through SCIP message handler */
      SmartPtr<Journal> jrnl = new ScipJournal("console", J_ITERSUMMARY, scip);
      jrnl->SetPrintLevel(J_DBG, J_NONE);
      if( !(*problem)->ipopt->Jnlst()->AddJournal(jrnl) )
      {
         SCIPerrorMessage("Failed to register ScipJournal for IPOPT output.");
      }

      /* initialize Ipopt/SCIP NLP interface */
      (*problem)->nlp = new ScipNLP(*problem, scip);
   }
   catch( const std::bad_alloc& )
   {
      SCIPerrorMessage("Not enough memory to initialize Ipopt.\n");
      return SCIP_NOMEMORY;
   }

   for( size_t i = 0; i < sizeof(ipopt_string_params) / sizeof(const char*); ++i )
   {
      SCIP_PARAM* param;
      char paramname[SCIP_MAXSTRLEN];
      char* paramval;

      strcpy(paramname, "nlpi/" NLPI_NAME "/");
      strcat(paramname, ipopt_string_params[i]);
      param = SCIPgetParam(scip, paramname);

      // skip parameters that we didn't add to SCIP because they didn't exist in this build of Ipopt
      if( param == NULL )
         continue;

      // if value wasn't left at the default, then pass to Ipopt and forbid overwriting
      paramval = SCIPparamGetString(param);
      assert(paramval != NULL);
      if( *paramval != '\0' )
         (void) (*problem)->ipopt->Options()->SetStringValue(ipopt_string_params[i], paramval, false);
   }

   for( size_t i = 0; i < sizeof(ipopt_int_params) / sizeof(const char*); ++i )
   {
      SCIP_PARAM* param;
      char paramname[SCIP_MAXSTRLEN];
      int paramval;

      strcpy(paramname, "nlpi/" NLPI_NAME "/");
      strcat(paramname, ipopt_int_params[i]);
      param = SCIPgetParam(scip, paramname);

      // skip parameters that we didn't add to SCIP because they didn't exist in this build of Ipopt
      if( param == NULL )
         continue;

      // if value wasn't left at the default, then pass to Ipopt and forbid overwriting
      paramval = SCIPparamGetInt(param);
      if( paramval != SCIPparamGetIntDefault(param) )
      {
         (void) (*problem)->ipopt->Options()->SetIntegerValue(ipopt_int_params[i], paramval, false);

         if( i == 0 )
         {
            assert(strcmp(ipopt_int_params[0], "print_level") == 0);
            (*problem)->printlevelset = true;
         }
      }
   }

#if defined(__GNUC__) && IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR < 14
   /* Turn off bound relaxation for older Ipopt, as solutions may be out of bounds by more than constr_viol_tol.
    * For Ipopt 3.14, bounds are relaxed by at most constr_viol_tol, so can leave bound_relax_factor at its default.
    */
   (void) (*problem)->ipopt->Options()->SetNumericValue("bound_relax_factor", 0.0);
#endif

   /* modify Ipopt's default settings to what we believe is appropriate */
   /* (*problem)->ipopt->Options()->SetStringValue("print_timing_statistics", "yes"); */
#ifdef SCIP_DEBUG
   (void) (*problem)->ipopt->Options()->SetStringValue("print_user_options", "yes");
#endif
   (void) (*problem)->ipopt->Options()->SetStringValue("sb", "yes");
   (void) (*problem)->ipopt->Options()->SetStringValueIfUnset("mu_strategy", "adaptive");
   (void) (*problem)->ipopt->Options()->SetIntegerValue("max_iter", INT_MAX);
   (void) (*problem)->ipopt->Options()->SetNumericValue("nlp_lower_bound_inf", -SCIPinfinity(scip), false);
   (void) (*problem)->ipopt->Options()->SetNumericValue("nlp_upper_bound_inf",  SCIPinfinity(scip), false);
   (void) (*problem)->ipopt->Options()->SetNumericValue("diverging_iterates_tol", SCIPinfinity(scip), false);

   /* apply user's given modifications to Ipopt's default settings */
   if( data->defoptions.length() > 0 )
   {
      std::istringstream is(data->defoptions);

      if( !(*problem)->ipopt->Options()->ReadFromStream(*(*problem)->ipopt->Jnlst(), is, true) )
      {
         SCIPerrorMessage("Error when modifying Ipopt options using options string\n%s\n", data->defoptions.c_str());
         return SCIP_ERROR;
      }
   }

   /* apply user's given options file */
   assert(data->optfile != NULL);
   if( (*problem)->ipopt->Initialize(data->optfile) != Solve_Succeeded )
   {
      SCIPerrorMessage("Error during initialization of Ipopt using optionfile \"%s\"\n", data->optfile);
      return SCIP_ERROR;
   }

   return SCIP_OKAY;
}

/** free a problem instance
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem pointer where problem data is stored 
 */
static
SCIP_DECL_NLPIFREEPROBLEM(nlpiFreeProblemIpopt)
{
   int n;
   int m;

   assert(nlpi     != NULL);
   assert(problem  != NULL);
   assert(*problem != NULL);
   assert((*problem)->oracle != NULL);

   n = SCIPnlpiOracleGetNVars((*problem)->oracle);
   m = SCIPnlpiOracleGetNConstraints((*problem)->oracle);

   SCIPfreeBlockMemoryArrayNull(scip, &(*problem)->solprimals, n);
   SCIPfreeBlockMemoryArrayNull(scip, &(*problem)->soldualcons, m);
   SCIPfreeBlockMemoryArrayNull(scip, &(*problem)->soldualvarlb, n);
   SCIPfreeBlockMemoryArrayNull(scip, &(*problem)->soldualvarub, n);

   SCIP_CALL( SCIPnlpiOracleFree(scip, &(*problem)->oracle) );

   if( (*problem)->randnumgen != NULL )
   {
      SCIPfreeRandom(scip, &(*problem)->randnumgen);
   }

   delete *problem;
   *problem = NULL;

   return SCIP_OKAY;
}

/** gets pointer to solver-internal problem instance to do dirty stuff
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *
 *  return: void pointer to problem instance
 */
static
SCIP_DECL_NLPIGETPROBLEMPOINTER(nlpiGetProblemPointerIpopt)
{
   assert(nlpi    != NULL);
   assert(problem != NULL);

   return GetRawPtr(problem->nlp);
}

/** add variables
 * 
 * input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - nvars number of variables 
 *  - lbs lower bounds of variables, can be NULL if -infinity
 *  - ubs upper bounds of variables, can be NULL if +infinity
 *  - varnames names of variables, can be NULL
 */
static
SCIP_DECL_NLPIADDVARS(nlpiAddVarsIpopt)
{
   int oldnvars;

   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);

   oldnvars = SCIPnlpiOracleGetNVars(problem->oracle);

   SCIPfreeBlockMemoryArrayNull(scip, &problem->solprimals, oldnvars);
   SCIPfreeBlockMemoryArrayNull(scip, &problem->soldualvarlb, oldnvars);
   SCIPfreeBlockMemoryArrayNull(scip, &problem->soldualvarub, oldnvars);
   invalidateSolution(problem);

   SCIP_CALL( SCIPnlpiOracleAddVars(scip, problem->oracle, nvars, lbs, ubs, varnames) );

   problem->samestructure = false;

   return SCIP_OKAY;
}

/** add constraints
 *
 * quadratic coefficiens: row oriented matrix for each constraint
 * 
 * input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - ncons number of added constraints
 *  - lhss left hand sides of constraints
 *  - rhss right hand sides of constraints
 *  - linoffsets start index of each constraints linear coefficients in lininds and linvals
 *    length: ncons + 1, linoffsets[ncons] gives length of lininds and linvals
 *    may be NULL in case of no linear part
 *  - lininds variable indices
 *    may be NULL in case of no linear part
 *  - linvals coefficient values
 *    may be NULL in case of no linear part
 *  - nquadelems number of quadratic elements for each constraint
 *    may be NULL in case of no quadratic part
 *  - quadelems quadratic elements for each constraint
 *    may be NULL in case of no quadratic part
 *  - exprvaridxs indices of variables in expression tree, maps variable indices in expression tree to indices in nlp
 *    entry of array may be NULL in case of no expression tree
 *    may be NULL in case of no expression tree in any constraint
 *  - exprtrees expression tree for nonquadratic part of constraints
 *    entry of array may be NULL in case of no nonquadratic part
 *    may be NULL in case of no nonquadratic part in any constraint
 *  - names of constraints, may be NULL or entries may be NULL
 */
static
SCIP_DECL_NLPIADDCONSTRAINTS(nlpiAddConstraintsIpopt)
{
   int oldncons;

   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);

   oldncons = SCIPnlpiOracleGetNConstraints(problem->oracle);

   SCIPfreeBlockMemoryArrayNull(scip, &problem->soldualcons, oldncons);
   problem->soldualvalid = false;
   problem->soldualgiven = false;

   SCIP_CALL( SCIPnlpiOracleAddConstraints(scip, problem->oracle, nconss, lhss, rhss, nlininds, lininds, linvals, exprs, names) );

   problem->samestructure = false;

   return SCIP_OKAY;
}

/** sets or overwrites objective, a minimization problem is expected
 *
 *  May change sparsity pattern.
 *
 * input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - nlins number of linear variables
 *  - lininds variable indices
 *    may be NULL in case of no linear part
 *  - linvals coefficient values
 *    may be NULL in case of no linear part
 *  - nquadelems number of elements in matrix of quadratic part
 *  - quadelems elements of quadratic part
 *    may be NULL in case of no quadratic part
 *  - exprvaridxs indices of variables in expression tree, maps variable indices in expression tree to indices in nlp
 *    may be NULL in case of no expression tree
 *  - exprtree expression tree for nonquadratic part of objective function
 *    may be NULL in case of no nonquadratic part
 *  - constant objective value offset
 */
static
SCIP_DECL_NLPISETOBJECTIVE(nlpiSetObjectiveIpopt)
{
   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);

   /* We pass the objective gradient in dense form to Ipopt, so if the sparsity of that gradient changes, we do not change the structure of the problem inside Ipopt.
    * However, if the sparsity of the Hessian matrix of the objective changes, then the sparsity pattern of the Hessian of the Lagrangian may change.
    * Thus, set samestructure=false if the objective was and/or becomes nonlinear, but leave samestructure untouched if it was and stays linear.
    */
   if( expr != NULL || SCIPnlpiOracleGetConstraintDegree(problem->oracle, -1) > 1 )
      problem->samestructure = false;

   SCIP_CALL( SCIPnlpiOracleSetObjective(scip, problem->oracle, constant, nlins, lininds, linvals, expr) );

   /* keep solution as valid, but reset solve status and objective value */
   invalidateSolved(problem);

   return SCIP_OKAY;
}

/** change variable bounds
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - nvars number of variables to change bounds
 *  - indices indices of variables to change bounds
 *  - lbs new lower bounds
 *  - ubs new upper bounds
 */
static
SCIP_DECL_NLPICHGVARBOUNDS(nlpiChgVarBoundsIpopt)
{
   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);

   /* Check whether the structure of the Ipopt internal NLP changes, if problem->samestructure at the moment.
    * We need to check whether variables become fixed or unfixed and whether bounds are added or removed.
    *
    * Project primal solution onto new bounds if currently valid.
    */
   if( problem->samestructure || problem->solprimalvalid )
   {
      for( int i = 0; i < nvars; ++i )
      {
         SCIP_Real oldlb;
         SCIP_Real oldub;
         oldlb = SCIPnlpiOracleGetVarLbs(problem->oracle)[indices[i]];
         oldub = SCIPnlpiOracleGetVarUbs(problem->oracle)[indices[i]];

         if( (oldlb == oldub) != (lbs[i] == ubs[i]) )  /*lint !e777*/
            problem->samestructure = false;
         else if( SCIPisInfinity(scip, -oldlb) != SCIPisInfinity(scip, -lbs[i]) )
            problem->samestructure = false;
         else if( SCIPisInfinity(scip,  oldub) != SCIPisInfinity(scip,  ubs[i]) )
            problem->samestructure = false;

         if( problem->solprimalvalid )
         {
            assert(problem->solprimals != NULL);
            problem->solprimals[i] = MIN(MAX(problem->solprimals[indices[i]], lbs[i]), ubs[i]);
         }
      }
   }

   SCIP_CALL( SCIPnlpiOracleChgVarBounds(scip, problem->oracle, nvars, indices, lbs, ubs) );

   invalidateSolved(problem);

   return SCIP_OKAY;
}

/** change constraint bounds
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - nconss number of constraints to change sides
 *  - indices indices of constraints to change sides
 *  - lhss new left hand sides
 *  - rhss new right hand sides
 */
static
SCIP_DECL_NLPICHGCONSSIDES(nlpiChgConsSidesIpopt)
{
   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);

   /* Check whether the structure of the Ipopt internal NLP changes, if problem->samestructure at the moment.
    * We need to check whether constraints change from equality to inequality and whether sides are added or removed.
    */
   for( int i = 0; i < nconss && problem->samestructure; ++i )
   {
      SCIP_Real oldlhs;
      SCIP_Real oldrhs;
      oldlhs = SCIPnlpiOracleGetConstraintLhs(problem->oracle, indices[i]);
      oldrhs = SCIPnlpiOracleGetConstraintRhs(problem->oracle, indices[i]);

      if( (oldlhs == oldrhs) != (lhss[i] == rhss[i]) )  /*lint !e777*/
         problem->samestructure = false;
      else if( SCIPisInfinity(scip, -oldlhs) != SCIPisInfinity(scip, -lhss[i]) )
         problem->samestructure = false;
      else if( SCIPisInfinity(scip,  oldrhs) != SCIPisInfinity(scip,  rhss[i]) )
         problem->samestructure = false;
   }

   SCIP_CALL( SCIPnlpiOracleChgConsSides(scip, problem->oracle, nconss, indices, lhss, rhss) );

   invalidateSolved(problem);

   return SCIP_OKAY;
}

/** delete a set of variables
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - nlpi datastructure for solver interface
 *  - dstats deletion status of vars; 1 if var should be deleted, 0 if not
 *
 *  output:
 *  - dstats new position of var, -1 if var was deleted
 */
static
SCIP_DECL_NLPIDELVARSET(nlpiDelVarSetIpopt)
{
   int nvars;

   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);
   assert(SCIPnlpiOracleGetNVars(problem->oracle) == dstatssize);

   SCIP_CALL( SCIPnlpiOracleDelVarSet(scip, problem->oracle, dstats) );

   nvars = SCIPnlpiOracleGetNVars(problem->oracle);

   if( problem->solprimalvalid || problem->soldualvalid )
   {
      // update existing solution, if valid
      assert(!problem->solprimalvalid || problem->solprimals != NULL);
      assert(!problem->soldualvalid || problem->soldualvarlb != NULL);
      assert(!problem->soldualvalid || problem->soldualvarub != NULL);

      int i;
      for( i = 0; i < dstatssize; ++i )
      {
         if( dstats[i] != -1 )
         {
            assert(dstats[i] >= 0);
            assert(dstats[i] < nvars);
            if( problem->solprimals != NULL )
               problem->solprimals[dstats[i]] = problem->solprimals[i];
            if( problem->soldualvarlb != NULL )
            {
               assert(problem->soldualvarub != NULL);
               problem->soldualvarlb[dstats[i]] = problem->soldualvarlb[i];
               problem->soldualvarub[dstats[i]] = problem->soldualvarub[i];
            }
         }
      }
   }

   /* resize solution point arrays */
   if( problem->solprimals != NULL )
   {
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &problem->solprimals, dstatssize, nvars) );
   }
   if( problem->soldualvarlb != NULL )
   {
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &problem->soldualvarlb, dstatssize, nvars) );
   }
   if( problem->soldualvarub != NULL )
   {
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &problem->soldualvarub, dstatssize, nvars) );
   }

   problem->samestructure = false;

   invalidateSolved(problem);

   return SCIP_OKAY;
}

/** delete a set of constraints
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - dstats deletion status of rows; 1 if row should be deleted, 0 if not
 *
 *  output:
 *  - dstats new position of row, -1 if row was deleted
 */
static
SCIP_DECL_NLPIDELCONSSET(nlpiDelConstraintSetIpopt)
{
   int ncons;

   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);
   assert(SCIPnlpiOracleGetNConstraints(problem->oracle) == dstatssize);

   SCIP_CALL( SCIPnlpiOracleDelConsSet(scip, problem->oracle, dstats) );

   ncons = SCIPnlpiOracleGetNConstraints(problem->oracle);

   if( problem->soldualvalid )
   {
      // update existing dual solution
      assert(problem->soldualcons != NULL);

      int i;
      for( i = 0; i < dstatssize; ++i )
      {
         if( dstats[i] != -1 )
         {
            assert(dstats[i] >= 0);
            assert(dstats[i] < ncons);
            problem->soldualcons[dstats[i]] = problem->soldualcons[i];
         }
      }
   }

   /* resize dual solution point array */
   if( problem->soldualcons != NULL )
   {
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &problem->soldualcons, dstatssize, ncons) );
   }

   problem->samestructure = false;

   invalidateSolved(problem);

   return SCIP_OKAY;
}

/** change one linear coefficient in a constraint or objective
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - idx index of constraint or -1 for objective
 *  - nvals number of values in linear constraint
 *  - varidxs indices of variable
 *  - vals new values for coefficient
 *
 *  return: Error if coefficient did not exist before
 */
static
SCIP_DECL_NLPICHGLINEARCOEFS(nlpiChgLinearCoefsIpopt)
{
   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);

   SCIP_CALL( SCIPnlpiOracleChgLinearCoefs(scip, problem->oracle, idx, nvals, varidxs, vals) );

   invalidateSolved(problem);

   return SCIP_OKAY;
}

/** replaces the expression tree of a constraint or objective
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - idxcons index of constraint or -1 for objective
 *  - exprtree new expression tree for constraint or objective, or NULL to only remove previous tree
 */
static
SCIP_DECL_NLPICHGEXPR(nlpiChgExprIpopt)
{
   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);

   SCIP_CALL( SCIPnlpiOracleChgExpr(scip, problem->oracle, idxcons, expr) );

   problem->samestructure = false;  // nonzero patterns may have changed
   invalidateSolved(problem);

   return SCIP_OKAY;
}

/** change the constant offset in the objective
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - objconstant new value for objective constant
 */
static
SCIP_DECL_NLPICHGOBJCONSTANT(nlpiChgObjConstantIpopt)
{
   SCIP_Real oldconstant;

   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);

   oldconstant = SCIPnlpiOracleGetObjectiveConstant(problem->oracle);

   SCIP_CALL( SCIPnlpiOracleChgObjConstant(scip, problem->oracle, objconstant) );

   if( problem->solobjval != SCIP_INVALID )
      problem->solobjval += objconstant - oldconstant;

   return SCIP_OKAY;
}

/** sets initial guess for primal variables
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - primalvalues initial primal values for variables, or NULL to clear previous values
 *  - consdualvalues initial dual values for constraints, or NULL to clear previous values
 *  - varlbdualvalues  initial dual values for variable lower bounds, or NULL to clear previous values
 *  - varubdualvalues  initial dual values for variable upper bounds, or NULL to clear previous values
 */
static
SCIP_DECL_NLPISETINITIALGUESS(nlpiSetInitialGuessIpopt)
{
   int nvars;

   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);

   nvars = SCIPnlpiOracleGetNVars(problem->oracle);

   if( primalvalues != NULL )
   {
      // copy primal solution
      SCIPdebugMsg(scip, "set initial guess primal values to user-given\n");
      if( problem->solprimals == NULL )
      {
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &problem->solprimals, nvars) );
      }
      BMScopyMemoryArray(problem->solprimals, primalvalues, nvars);
      problem->solprimalvalid = true;
      problem->solprimalgiven = true;
   }
   else
   {
      // invalid current primal solution (if any)
      if( problem->solprimalvalid )
      {
         SCIPdebugMsg(scip, "invalidate initial guess primal values on user-request\n");
      }
      problem->solprimalvalid = false;
      problem->solprimalgiven = false;
   }

   if( consdualvalues != NULL && varlbdualvalues != NULL && varubdualvalues != NULL )
   {
      // copy dual solution, if completely given
      SCIPdebugMsg(scip, "set initial guess dual values to user-given\n");
      int ncons = SCIPnlpiOracleGetNConstraints(problem->oracle);
      if( problem->soldualcons == NULL )
      {
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &problem->soldualcons, ncons) );
      }
      BMScopyMemoryArray(problem->soldualcons, consdualvalues, ncons);

      assert((problem->soldualvarlb == NULL) == (problem->soldualvarub == NULL));
      if( problem->soldualvarlb != NULL )
      {
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &problem->soldualvarlb, nvars) );
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &problem->soldualvarub, nvars) );
      }
      BMScopyMemoryArray(problem->soldualvarlb, varlbdualvalues, nvars);
      BMScopyMemoryArray(problem->soldualvarub, varubdualvalues, nvars);
      problem->soldualvalid = true;
      problem->soldualgiven = true;
   }
   else
   {
      // invalid current dual solution (if any)
      if( problem->soldualvalid )
      {
         SCIPdebugMsg(scip, "invalidate initial guess dual values\n");
      }
      problem->soldualvalid = false;
      problem->soldualgiven = false;
   }

   return SCIP_OKAY;
}

/** tries to solve NLP
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 */
static
SCIP_DECL_NLPISOLVE(nlpiSolveIpopt)
{
   ApplicationReturnStatus status;

   assert(nlpi != NULL);
   assert(problem != NULL);
   assert(problem->oracle != NULL);

   assert(IsValid(problem->ipopt));
   assert(IsValid(problem->nlp));

   SCIPdebugMsg(scip, "solve with parameters " SCIP_NLPPARAM_PRINT(param));

   if( param.timelimit == 0.0 )
   {
      /* there is nothing we can do if we are not given any time */
      problem->lastniter = 0;
      problem->lasttime = 0.0;
      problem->termstat = SCIP_NLPTERMSTAT_TIMELIMIT;
      problem->solstat = SCIP_NLPSOLSTAT_UNKNOWN;

      return SCIP_OKAY;
   }

   // change status info to unsolved, just in case
   invalidateSolved(problem);

   // ensure a starting point is available
   // also disables param.warmstart if no warmstart available
   SCIP_CALL( ensureStartingPoint(scip, problem, param.warmstart) );

   // tell NLP that we are about to start a new solve
   problem->nlp->initializeSolve(problem, param);

   // set Ipopt parameters
   SCIP_CALL( handleNlpParam(scip, problem, param) );

   try
   {
      SmartPtr<SolveStatistics> stats;

#ifdef PROTECT_SOLVE_BY_MUTEX
      /* lock solve_mutex if Ipopt is going to use Mumps as linear solver
       * unlocking will happen in the destructor of guard, which is called when this block is left
       */
      std::unique_lock<std::mutex> guard(solve_mutex, std::defer_lock);  /*lint !e{728}*/
      std::string linsolver;
      (void) problem->ipopt->Options()->GetStringValue("linear_solver", linsolver, "");
      if( linsolver == "mumps" )
         guard.lock();
#endif

      if( problem->firstrun )
      {
         SCIP_EXPRINTCAPABILITY cap;

         cap = SCIPexprintGetCapability() & SCIPnlpiOracleGetEvalCapability(scip, problem->oracle);

         /* if the expression interpreter or some user expression do not support function values and gradients and Hessians,
          * change NLP parameters or give an error
          */
         if( (cap & (SCIP_EXPRINTCAPABILITY_FUNCVALUE | SCIP_EXPRINTCAPABILITY_GRADIENT | SCIP_EXPRINTCAPABILITY_HESSIAN)) != (SCIP_EXPRINTCAPABILITY_FUNCVALUE | SCIP_EXPRINTCAPABILITY_GRADIENT | SCIP_EXPRINTCAPABILITY_HESSIAN) )
         {
            if( !(SCIPexprintGetCapability() & SCIP_EXPRINTCAPABILITY_FUNCVALUE) ||
                !(SCIPexprintGetCapability() & SCIP_EXPRINTCAPABILITY_GRADIENT) )
            {
               SCIPerrorMessage("Do not have expression interpreter that can compute function values and gradients. Cannot solve NLP with Ipopt.\n");
               problem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
               problem->termstat = SCIP_NLPTERMSTAT_OTHER;
               return SCIP_OKAY;
            }

            /* enable Hessian approximation if we are nonquadratic and the expression interpreter or user expression do not support Hessians */
            if( !(cap & SCIP_EXPRINTCAPABILITY_HESSIAN) )
            {
               (void) problem->ipopt->Options()->SetStringValueIfUnset("hessian_approximation", "limited-memory");
               problem->nlp->approxhessian = true;
            }
            else
               problem->nlp->approxhessian = false;
         }

#ifdef SCIP_DEBUG
         problem->ipopt->Options()->SetStringValue("derivative_test", problem->nlp->approxhessian ? "first-order" : "second-order");
#endif

         status = problem->ipopt->OptimizeTNLP(GetRawPtr(problem->nlp));
      }
      else
      {
         // TODO to be strict, we should check whether the eval capability has been changed and the Hessian approximation needs to be enabled (in which case we should call OptimizeTNLP instead)
         problem->ipopt->Options()->SetStringValue("warm_start_same_structure", problem->samestructure ? "yes" : "no");
         status = problem->ipopt->ReOptimizeTNLP(GetRawPtr(problem->nlp));
      }

      // catch the very bad status codes
      switch( status ) {
         // everything better than Not_Enough_Degrees_Of_Freedom is a non-serious error
         case Solve_Succeeded:
         case Solved_To_Acceptable_Level:
         case Infeasible_Problem_Detected:
         case Search_Direction_Becomes_Too_Small:
         case Diverging_Iterates:
         case User_Requested_Stop:
         case Feasible_Point_Found:
         case Maximum_Iterations_Exceeded:
         case Restoration_Failed:
         case Error_In_Step_Computation:
         case Maximum_CpuTime_Exceeded:
#if IPOPT_VERSION_MAJOR > 3 || IPOPT_VERSION_MINOR >= 14
         case Maximum_WallTime_Exceeded:   // new in Ipopt 3.14
            // if Ipopt >= 3.14, finalize_solution should always have been called if we get these status codes
            // this should have left us with some solution (unless we ran out of memory in finalize_solution)
            assert(problem->solprimalvalid || problem->termstat == SCIP_NLPTERMSTAT_OUTOFMEMORY);
            assert(problem->soldualvalid || problem->termstat == SCIP_NLPTERMSTAT_OUTOFMEMORY);
#endif
            problem->firstrun = false;
            problem->samestructure = true;
            break;

         case Not_Enough_Degrees_Of_Freedom:
            assert(problem->termstat == SCIP_NLPTERMSTAT_OTHER);
            assert(problem->solstat == SCIP_NLPSOLSTAT_UNKNOWN);
            SCIPdebugMsg(scip, "NLP has too few degrees of freedom.\n");
            break;

         case Invalid_Number_Detected:
            SCIPdebugMsg(scip, "Ipopt failed because of an invalid number in function or derivative value\n");
            problem->termstat = SCIP_NLPTERMSTAT_EVALERROR;
            assert(problem->solstat == SCIP_NLPSOLSTAT_UNKNOWN);
            break;

         case Insufficient_Memory:
            assert(problem->termstat == SCIP_NLPTERMSTAT_OTHER);
            assert(problem->solstat == SCIP_NLPSOLSTAT_UNKNOWN);
            SCIPerrorMessage("Ipopt returned with status \"Insufficient Memory\"\n");
            return SCIP_NOMEMORY;

         // the really bad ones that indicate rather a programming error
         case Invalid_Problem_Definition:
         case Invalid_Option:
         case Unrecoverable_Exception:
         case NonIpopt_Exception_Thrown:
         case Internal_Error:
            assert(problem->termstat == SCIP_NLPTERMSTAT_OTHER);
            assert(problem->solstat == SCIP_NLPSOLSTAT_UNKNOWN);
            SCIPerrorMessage("Ipopt returned with application return status %d\n", status);
            return SCIP_ERROR;
      }

      stats = problem->ipopt->Statistics();
      if( IsValid(stats) )
      {
         problem->lastniter = stats->IterationCount();
         problem->lasttime  = stats->TotalWallclockTime();
      }
      else
      {
         /* Ipopt does not provide access to the statistics if there was a serious error */
         problem->lastniter = 0;
         problem->lasttime  = 0.0;
      }
   }
   catch( IpoptException& except )
   {
      SCIPerrorMessage("Ipopt returned with exception: %s\n", except.Message().c_str());
      return SCIP_ERROR;
   }

   return SCIP_OKAY;
}

/** gives solution status
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *
 *  return: Solution Status
 */
static
SCIP_DECL_NLPIGETSOLSTAT(nlpiGetSolstatIpopt)
{
   assert(nlpi != NULL);
   assert(problem != NULL);

   return problem->solstat;
}

/** gives termination reason
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *
 *  return: Termination Status
 */
static
SCIP_DECL_NLPIGETTERMSTAT(nlpiGetTermstatIpopt)
{
   assert(nlpi != NULL);
   assert(problem != NULL);

   return problem->termstat;
}

/** gives primal and dual solution values
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - primalvalues buffer to store pointer to array to primal values, or NULL if not needed
 *  - consdualvalues buffer to store pointer to array to dual values of constraints, or NULL if not needed
 *  - varlbdualvalues buffer to store pointer to array to dual values of variable lower bounds, or NULL if not needed
 *  - varubdualvalues buffer to store pointer to array to dual values of variable lower bounds, or NULL if not needed
 *  - objval buffer store the objective value, or NULL if not needed
 */
static
SCIP_DECL_NLPIGETSOLUTION(nlpiGetSolutionIpopt)
{
   assert(nlpi != NULL);
   assert(problem != NULL);

   if( primalvalues != NULL )
      *primalvalues = problem->solprimals;

   if( consdualvalues != NULL )
      *consdualvalues = problem->soldualcons;

   if( varlbdualvalues != NULL )
      *varlbdualvalues = problem->soldualvarlb;

   if( varubdualvalues != NULL )
      *varubdualvalues = problem->soldualvarub;

   if( objval != NULL )
      *objval = problem->solobjval;

   return SCIP_OKAY;
}

/** gives solve statistics
 *
 *  input:
 *  - nlpi datastructure for solver interface
 *  - problem datastructure for problem instance
 *  - statistics pointer to store statistics
 *
 *  output:
 *  - statistics solve statistics
 */
static
SCIP_DECL_NLPIGETSTATISTICS(nlpiGetStatisticsIpopt)
{
   assert(nlpi != NULL);
   assert(problem != NULL);

   SCIPnlpStatisticsSetNIterations(statistics, problem->lastniter);
   SCIPnlpStatisticsSetTotalTime  (statistics, problem->lasttime);

   return SCIP_OKAY;
}

/** create solver interface for Ipopt solver and includes it into SCIP, if Ipopt is available */
SCIP_RETCODE SCIPincludeNlpSolverIpopt(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_NLPIDATA* nlpidata;
   SCIP_Bool advanced = FALSE;

   assert(scip != NULL);

   SCIP_ALLOC( nlpidata = new SCIP_NLPIDATA() ); /*lint !e774*/

   SCIP_CALL( SCIPincludeNlpi(scip,
         NLPI_NAME, NLPI_DESC, NLPI_PRIORITY,
         nlpiCopyIpopt, nlpiFreeIpopt, nlpiGetSolverPointerIpopt,
         nlpiCreateProblemIpopt, nlpiFreeProblemIpopt, nlpiGetProblemPointerIpopt,
         nlpiAddVarsIpopt, nlpiAddConstraintsIpopt, nlpiSetObjectiveIpopt,
         nlpiChgVarBoundsIpopt, nlpiChgConsSidesIpopt, nlpiDelVarSetIpopt, nlpiDelConstraintSetIpopt,
         nlpiChgLinearCoefsIpopt, nlpiChgExprIpopt,
         nlpiChgObjConstantIpopt, nlpiSetInitialGuessIpopt, nlpiSolveIpopt, nlpiGetSolstatIpopt, nlpiGetTermstatIpopt,
         nlpiGetSolutionIpopt, nlpiGetStatisticsIpopt,
         nlpidata) );

   SCIP_CALL( SCIPincludeExternalCodeInformation(scip, SCIPgetSolverNameIpopt(), SCIPgetSolverDescIpopt()) );

   SCIP_CALL( SCIPaddStringParam(scip, "nlpi/" NLPI_NAME "/optfile", "name of Ipopt options file",
      &nlpidata->optfile, FALSE, "", NULL, NULL) );

   SmartPtr<RegisteredOptions> reg_options = new RegisteredOptions();
   IpoptApplication::RegisterAllIpoptOptions(reg_options);

   for( size_t i = 0; i < sizeof(ipopt_string_params) / sizeof(const char*); ++i )
   {
      SmartPtr<const RegisteredOption> option = reg_options->GetOption(ipopt_string_params[i]);

      // skip options not available with this build of Ipopt
      if( !IsValid(option) )
         continue;

      assert(option->Type() == OT_String);

      // prefix parameter name with nlpi/ipopt
      std::string paramname("nlpi/" NLPI_NAME "/");
      paramname += option->Name();

      // initialize description with short description from Ipopt
      std::stringstream descr;
      descr << option->ShortDescription();

      // add valid values to description, if there are more than one
      // the only case where there are less than 2 valid strings should be when anything is valid (in which case there is one valid string with value "*")
      std::vector<RegisteredOption::string_entry> validvals = option->GetValidStrings();
      if( validvals.size() > 1 )
      {
         descr << " Valid values if not empty:";
         for( std::vector<RegisteredOption::string_entry>::iterator val = validvals.begin(); val != validvals.end(); ++val )
            descr << ' ' << val->value_;
      }

#if IPOPT_VERSION_MAJOR > 3 || IPOPT_VERSION_MINOR >= 14
      // since Ipopt 3.14, Ipopt options have an advanced flag
      advanced = option->Advanced();
#endif

      // we use the empty string as default to recognize later whether the user set has set the option
      SCIP_CALL( SCIPaddStringParam(scip, paramname.c_str(), descr.str().c_str(), NULL, advanced, "", NULL, NULL) );
   }

   for( size_t i = 0; i < sizeof(ipopt_int_params) / sizeof(const char*); ++i )
   {
      SmartPtr<const RegisteredOption> option = reg_options->GetOption(ipopt_int_params[i]);

      // skip options not available with this build of Ipopt
      if( !IsValid(option) )
         continue;

      assert(option->Type() == OT_Integer);

      // prefix parameter name with nlpi/ipopt
      std::string paramname("nlpi/" NLPI_NAME "/");
      paramname += option->Name();

      int lower = option->LowerInteger();
      int upper = option->UpperInteger();

      // we use value lower-1 as signal that the option was not modified by the user
      // for that, we require a finite lower bound
      assert(lower > INT_MIN);

      // initialize description with short description from Ipopt
      std::stringstream descr;
      descr << option->ShortDescription();
      descr << ' ' << (lower-1) << " to use NLPI or Ipopt default.";

#if IPOPT_VERSION_MAJOR > 3 || IPOPT_VERSION_MINOR >= 14
      // since Ipopt 3.14, Ipopt options have an advanced flag
      advanced = option->Advanced();
#endif

      // we use the empty string as default to recognize later whether the user set has set the option
      SCIP_CALL( SCIPaddIntParam(scip, paramname.c_str(), descr.str().c_str(), NULL, advanced, lower-1, lower-1, upper, NULL, NULL) );
   }

   return SCIP_OKAY;
}  /*lint !e429 */

/** gets string that identifies Ipopt (version number) */
const char* SCIPgetSolverNameIpopt(void)
{
   return "Ipopt " IPOPT_VERSION;
}

/** gets string that describes Ipopt */
const char* SCIPgetSolverDescIpopt(void)
{
   return "Interior Point Optimizer developed by A. Waechter et.al. (github.com/coin-or/Ipopt)";
}

/** returns whether Ipopt is available, i.e., whether it has been linked in */
SCIP_Bool SCIPisIpoptAvailableIpopt(void)
{
   return TRUE;
}

/** gives a pointer to the IpoptApplication object stored in Ipopt-NLPI's NLPI problem data structure */
void* SCIPgetIpoptApplicationPointerIpopt(
   SCIP_NLPIPROBLEM*     nlpiproblem         /**< NLP problem of Ipopt-NLPI */
   )
{
   assert(nlpiproblem != NULL);

   return (void*)GetRawPtr(nlpiproblem->ipopt);
}

/** gives a pointer to the NLPIORACLE object stored in Ipopt-NLPI's NLPI problem data structure */
void* SCIPgetNlpiOracleIpopt(
   SCIP_NLPIPROBLEM*     nlpiproblem         /**< NLP problem of Ipopt-NLPI */
   )
{
   assert(nlpiproblem != NULL);

   return nlpiproblem->oracle;
}

/** sets modified default settings that are used when setting up an Ipopt problem
 *
 *  Do not forget to add a newline after the last option in optionsstring.
 */
void SCIPsetModifiedDefaultSettingsIpopt(
   SCIP_NLPI*            nlpi,               /**< Ipopt NLP interface */
   const char*           optionsstring,      /**< string with options as in Ipopt options file */
   SCIP_Bool             append              /**< whether to append to modified default settings or to overwrite */
   )
{
   SCIP_NLPIDATA* data;

   assert(nlpi != NULL);

   data = SCIPnlpiGetData(nlpi);
   assert(data != NULL);

   if( append )
      data->defoptions += optionsstring;
   else
      data->defoptions = optionsstring;
}

/** Method to return some info about the nlp */
bool ScipNLP::get_nlp_info(
   Index&             n,                  /**< place to store number of variables */ 
   Index&             m,                  /**< place to store number of constraints */ 
   Index&             nnz_jac_g,          /**< place to store number of nonzeros in jacobian */
   Index&             nnz_h_lag,          /**< place to store number of nonzeros in hessian */
   IndexStyleEnum&    index_style         /**< place to store used index style (0-based or 1-based) */
   )
{
   const int* offset;
   SCIP_RETCODE retcode;

   assert(nlpiproblem != NULL);
   assert(nlpiproblem->oracle != NULL);

   n = SCIPnlpiOracleGetNVars(nlpiproblem->oracle);
   m = SCIPnlpiOracleGetNConstraints(nlpiproblem->oracle);

   retcode = SCIPnlpiOracleGetJacobianSparsity(scip, nlpiproblem->oracle, &offset, NULL);
   if( retcode != SCIP_OKAY )
      return false;
   assert(offset != NULL);
   nnz_jac_g = offset[m];

   if( !approxhessian )
   {
      retcode = SCIPnlpiOracleGetHessianLagSparsity(scip, nlpiproblem->oracle, &offset, NULL);
      if( retcode != SCIP_OKAY )
         return false;
      assert(offset != NULL);
      nnz_h_lag = offset[n];
   }
   else
   {
      nnz_h_lag = 0;
   }

   index_style = TNLP::C_STYLE;

   return true;
}

/** Method to return the bounds for my problem */
bool ScipNLP::get_bounds_info(
   Index              n,                  /**< number of variables */ 
   Number*            x_l,                /**< buffer to store lower bounds on variables */
   Number*            x_u,                /**< buffer to store upper bounds on variables */
   Index              m,                  /**< number of constraints */
   Number*            g_l,                /**< buffer to store lower bounds on constraints */
   Number*            g_u                 /**< buffer to store lower bounds on constraints */
   )
{
   assert(nlpiproblem != NULL);
   assert(nlpiproblem->oracle != NULL);

   assert(n == SCIPnlpiOracleGetNVars(nlpiproblem->oracle));
   assert(m == SCIPnlpiOracleGetNConstraints(nlpiproblem->oracle));

   assert(SCIPnlpiOracleGetVarLbs(nlpiproblem->oracle) != NULL || n == 0);
   assert(SCIPnlpiOracleGetVarUbs(nlpiproblem->oracle) != NULL || n == 0);

   BMScopyMemoryArray(x_l, SCIPnlpiOracleGetVarLbs(nlpiproblem->oracle), n);
   BMScopyMemoryArray(x_u, SCIPnlpiOracleGetVarUbs(nlpiproblem->oracle), n);
#ifndef NDEBUG
   for( int i = 0; i < n; ++i )
      assert(x_l[i] <= x_u[i]);
#endif

   /* Ipopt performs better when unused variables do not appear, which we can achieve by fixing them,
    * since Ipopts TNLPAdapter will hide them from Ipopts NLP. In the dual solution, bound multipliers (z_L, z_U)
    * for these variables should have value 0.0 (they are set to -grad Lagrangian).
    */
   for( int i = 0; i < n; ++i )
   {
      int vardegree;
      if( SCIPnlpiOracleGetVarDegree(scip, nlpiproblem->oracle, i, &vardegree) != SCIP_OKAY )
         return false;
      if( vardegree == 0 )
      {
         SCIPdebugMsg(scip, "fix unused variable x%d [%g,%g] to 0.0 or bound\n", i, x_l[i], x_u[i]);
         assert(x_l[i] <= x_u[i]);
         x_l[i] = x_u[i] = MAX(MIN(x_u[i], 0.0), x_l[i]);
      }
   }

   for( int i = 0; i < m; ++i )
   {
      g_l[i] = SCIPnlpiOracleGetConstraintLhs(nlpiproblem->oracle, i);
      g_u[i] = SCIPnlpiOracleGetConstraintRhs(nlpiproblem->oracle, i);
      assert(g_l[i] <= g_u[i]);
   }

   return true;
}

/** Method to return the starting point for the algorithm */  /*lint -e{715}*/
bool ScipNLP::get_starting_point(
   Index              n,                  /**< number of variables */ 
   bool               init_x,             /**< whether initial values for primal values are requested */ 
   Number*            x,                  /**< buffer to store initial primal values */
   bool               init_z,             /**< whether initial values for dual values of variable bounds are requested */  
   Number*            z_L,                /**< buffer to store dual values for variable lower bounds */
   Number*            z_U,                /**< buffer to store dual values for variable upper bounds */
   Index              m,                  /**< number of constraints */
   bool               init_lambda,        /**< whether initial values for dual values of constraints are required */
   Number*            lambda              /**< buffer to store dual values of constraints */
   )
{  /*lint --e{715} */
   assert(nlpiproblem != NULL);
   assert(nlpiproblem->oracle != NULL);

   assert(n == SCIPnlpiOracleGetNVars(nlpiproblem->oracle));
   assert(m == SCIPnlpiOracleGetNConstraints(nlpiproblem->oracle));

   if( init_x )
   {
      assert(nlpiproblem->solprimalvalid);
      assert(nlpiproblem->solprimals != NULL);
      BMScopyMemoryArray(x, nlpiproblem->solprimals, n);
   }

   if( init_z )
   {
      assert(nlpiproblem->soldualvalid);
      assert(nlpiproblem->soldualvarlb != NULL);
      assert(nlpiproblem->soldualvarub != NULL);
      BMScopyMemoryArray(z_L, nlpiproblem->soldualvarlb, n);
      BMScopyMemoryArray(z_U, nlpiproblem->soldualvarub, n);
   }

   if( init_lambda )
   {
      assert(nlpiproblem->soldualvalid);
      assert(nlpiproblem->soldualcons != NULL);
      BMScopyMemoryArray(lambda, nlpiproblem->soldualcons, m);
   }

   return true;
}

/** Method to return the number of nonlinear variables. */
Index ScipNLP::get_number_of_nonlinear_variables()
{
   int count;
   int n;

   assert(nlpiproblem != NULL);
   assert(nlpiproblem->oracle != NULL);

   n = SCIPnlpiOracleGetNVars(nlpiproblem->oracle);

   count = 0;
   for( int i = 0; i < n; ++i )
   {
      int vardegree;
      if( SCIPnlpiOracleGetVarDegree(scip, nlpiproblem->oracle, i, &vardegree) != SCIP_OKAY )
         return -1;  // this will make Ipopt assume that all variables are nonlinear, which I guess is ok if we got an error here
      if( vardegree > 1 )
         ++count;
   }

   return count;
}

/** Method to return the indices of the nonlinear variables */
bool ScipNLP::get_list_of_nonlinear_variables(
   Index              num_nonlin_vars,    /**< number of nonlinear variables */
   Index*             pos_nonlin_vars     /**< array to fill with indices of nonlinear variables */
   )
{
   int count;
   int n;

   assert(nlpiproblem != NULL);
   assert(nlpiproblem->oracle != NULL);

   n = SCIPnlpiOracleGetNVars(nlpiproblem->oracle);

   count = 0;
   for( int i = 0; i < n; ++i )
   {
      int vardegree;
      if( SCIPnlpiOracleGetVarDegree(scip, nlpiproblem->oracle, i, &vardegree) != SCIP_OKAY )
         return false;
      if( vardegree > 1 )
      {
         assert(count < num_nonlin_vars);
         pos_nonlin_vars[count++] = i;
      }
   }

   assert(count == num_nonlin_vars);

   return true;
}

/** Method to return metadata about variables and constraints */  /*lint -e{715}*/
bool ScipNLP::get_var_con_metadata(
   Index              n,                  /**< number of variables */
   StringMetaDataMapType& var_string_md,  /**< variable meta data of string type */
   IntegerMetaDataMapType& var_integer_md,/**< variable meta data of integer type */
   NumericMetaDataMapType& var_numeric_md,/**< variable meta data of numeric type */
   Index              m,                  /**< number of constraints */
   StringMetaDataMapType& con_string_md,  /**< constraint meta data of string type */
   IntegerMetaDataMapType& con_integer_md,/**< constraint meta data of integer type */
   NumericMetaDataMapType& con_numeric_md /**< constraint meta data of numeric type */
   )
{ /*lint --e{715}*/
   assert(nlpiproblem != NULL);
   assert(nlpiproblem->oracle != NULL);
   assert(n == SCIPnlpiOracleGetNVars(nlpiproblem->oracle));
   assert(m == SCIPnlpiOracleGetNConstraints(nlpiproblem->oracle));

   char** varnames = SCIPnlpiOracleGetVarNames(nlpiproblem->oracle);
   if( varnames != NULL )
   {
      std::vector<std::string>& varnamesvec(var_string_md["idx_names"]);
      varnamesvec.reserve((size_t)n);
      for( int i = 0; i < n; ++i )
      {
         if( varnames[i] != NULL )
         {
            varnamesvec.push_back(varnames[i]);  /*lint !e3701*/
         }
         else
         {
            char buffer[20];
            (void) sprintf(buffer, "nlpivar%8d", i);
            varnamesvec.push_back(buffer);
         }
      }
   }

   std::vector<std::string>& consnamesvec(con_string_md["idx_names"]);
   consnamesvec.reserve((size_t)m);
   for( int i = 0; i < m; ++i )
   {
      if( SCIPnlpiOracleGetConstraintName(nlpiproblem->oracle, i) != NULL )
      {
         consnamesvec.push_back(SCIPnlpiOracleGetConstraintName(nlpiproblem->oracle, i));
      }
      else
      {
         char buffer[20];
         (void) sprintf(buffer, "nlpicons%8d", i);
         consnamesvec.push_back(buffer);  /*lint !e3701*/
      }
   }

   return true;
}

/** Method to return the objective value */  /*lint -e{715}*/
bool ScipNLP::eval_f(
   Index              n,                  /**< number of variables */ 
   const Number*      x,                  /**< point to evaluate */ 
   bool               new_x,              /**< whether some function evaluation method has been called for this point before */
   Number&            obj_value           /**< place to store objective function value */
   )
{ /*lint --e{715}*/
   assert(nlpiproblem != NULL);
   assert(nlpiproblem->oracle != NULL);

   assert(n == SCIPnlpiOracleGetNVars(nlpiproblem->oracle));

   if( new_x )
      ++current_x;
   last_f_eval_x = current_x;

   return SCIPnlpiOracleEvalObjectiveValue(scip, nlpiproblem->oracle, x, &obj_value) == SCIP_OKAY;
}

/** Method to return the gradient of the objective */  /*lint -e{715}*/
bool ScipNLP::eval_grad_f(
   Index              n,                  /**< number of variables */ 
   const Number*      x,                  /**< point to evaluate */ 
   bool               new_x,              /**< whether some function evaluation method has been called for this point before */
   Number*            grad_f              /**< buffer to store objective gradient */
   )
{ /*lint --e{715}*/
   SCIP_Real dummy;

   assert(nlpiproblem != NULL);
   assert(nlpiproblem->oracle != NULL);

   assert(n == SCIPnlpiOracleGetNVars(nlpiproblem->oracle));

   if( new_x )
      ++current_x;
   else
   {
      // pass new_x = TRUE to objective gradient eval iff we have not evaluated the objective function at this point yet
      new_x = last_f_eval_x < current_x;
   }
   // if we evaluate the objective gradient with new_x = true, then this will also evaluate the objective function
   // (and if we do with new_x = false, then we already have last_f_eval_x == current_x anyway)
   last_f_eval_x = current_x;

   return SCIPnlpiOracleEvalObjectiveGradient(scip, nlpiproblem->oracle, x, new_x, &dummy, grad_f) == SCIP_OKAY;
}

/** Method to return the constraint residuals */  /*lint -e{715}*/
bool ScipNLP::eval_g(
   Index              n,                  /**< number of variables */ 
   const Number*      x,                  /**< point to evaluate */ 
   bool               new_x,              /**< whether some function evaluation method has been called for this point before */
   Index              m,                  /**< number of constraints */
   Number*            g                   /**< buffer to store constraint function values */
   )
{ /*lint --e{715}*/
   assert(nlpiproblem != NULL);
   assert(nlpiproblem->oracle != NULL);

   assert(n == SCIPnlpiOracleGetNVars(nlpiproblem->oracle));

   if( new_x )
      ++current_x;
   last_g_eval_x = current_x;

   return SCIPnlpiOracleEvalConstraintValues(scip, nlpiproblem->oracle, x, g) == SCIP_OKAY;
}

/** Method to return:
 *   1) The structure of the jacobian (if "values" is NULL)
 *   2) The values of the jacobian (if "values" is not NULL)
 */  /*lint -e{715}*/
bool ScipNLP::eval_jac_g(
   Index              n,                  /**< number of variables */ 
   const Number*      x,                  /**< point to evaluate */ 
   bool               new_x,              /**< whether some function evaluation method has been called for this point before */
   Index              m,                  /**< number of constraints */
   Index              nele_jac,           /**< number of nonzero entries in jacobian */ 
   Index*             iRow,               /**< buffer to store row indices of nonzero jacobian entries, or NULL if values are requested */
   Index*             jCol,               /**< buffer to store column indices of nonzero jacobian entries, or NULL if values are requested */                  
   Number*            values              /**< buffer to store values of nonzero jacobian entries, or NULL if structure is requested */
   )
{ /*lint --e{715}*/
   assert(nlpiproblem != NULL);
   assert(nlpiproblem->oracle != NULL);

   assert(n == SCIPnlpiOracleGetNVars(nlpiproblem->oracle));
   assert(m == SCIPnlpiOracleGetNConstraints(nlpiproblem->oracle));

   if( values == NULL )
   { /* Ipopt wants to know sparsity structure */
      const int* jacoffset;
      const int* jaccol;
      int j;
      int i;

      assert(iRow != NULL);
      assert(jCol != NULL);

      if( SCIPnlpiOracleGetJacobianSparsity(scip, nlpiproblem->oracle, &jacoffset, &jaccol) != SCIP_OKAY )
         return false;

      assert(jacoffset[0] == 0);
      assert(jacoffset[m] == nele_jac);
      j = jacoffset[0];
      for( i = 0; i < m; ++i )
         for( ; j < jacoffset[i+1]; ++j )
            iRow[j] = i;

      BMScopyMemoryArray(jCol, jaccol, nele_jac);
   }
   else
   {
      if( new_x )
         ++current_x;
      else
      {
         // pass new_x = TRUE to Jacobian eval iff we have not evaluated the constraint functions at this point yet
         new_x = last_g_eval_x < current_x;
      }
      // if we evaluate the Jacobian with new_x = true, then this will also evaluate the constraint functions
      // (and if we do with new_x = false, then we already have last_g_eval_x == current_x anyway)
      last_f_eval_x = current_x;

      if( SCIPnlpiOracleEvalJacobian(scip, nlpiproblem->oracle, x, new_x, NULL, values) != SCIP_OKAY )
         return false;
   }

   return true;
}

/** Method to return:
 *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
 *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
 */   /*lint -e{715}*/
bool ScipNLP::eval_h(
   Index              n,                  /**< number of variables */ 
   const Number*      x,                  /**< point to evaluate */ 
   bool               new_x,              /**< whether some function evaluation method has been called for this point before */
   Number             obj_factor,         /**< weight for objective function */ 
   Index              m,                  /**< number of constraints */
   const Number*      lambda,             /**< weights for constraint functions */ 
   bool               new_lambda,         /**< whether the hessian has been evaluated for these values of lambda before */
   Index              nele_hess,          /**< number of nonzero entries in hessian */
   Index*             iRow,               /**< buffer to store row indices of nonzero hessian entries, or NULL if values are requested */
   Index*             jCol,               /**< buffer to store column indices of nonzero hessian entries, or NULL if values are requested */                  
   Number*            values              /**< buffer to store values of nonzero hessian entries, or NULL if structure is requested */
   )
{  /*lint --e{715}*/
   assert(nlpiproblem != NULL);
   assert(nlpiproblem->oracle != NULL);

   assert(n == SCIPnlpiOracleGetNVars(nlpiproblem->oracle));
   assert(m == SCIPnlpiOracleGetNConstraints(nlpiproblem->oracle));

   if( values == NULL )
   { /* Ipopt wants to know sparsity structure */
      const int* heslagoffset;
      const int* heslagcol;
      int j;
      int i;

      assert(iRow != NULL);
      assert(jCol != NULL);

      if( SCIPnlpiOracleGetHessianLagSparsity(scip, nlpiproblem->oracle, &heslagoffset, &heslagcol) != SCIP_OKAY )
         return false;

      assert(heslagoffset[0] == 0);
      assert(heslagoffset[n] == nele_hess);
      j = heslagoffset[0];
      for( i = 0; i < n; ++i )
         for( ; j < heslagoffset[i+1]; ++j )
            iRow[j] = i;

      BMScopyMemoryArray(jCol, heslagcol, nele_hess);
   }
   else
   {
      bool new_x_obj = new_x;
      bool new_x_cons = new_x;
      if( new_x )
         ++current_x;
      else
      {
         // pass new_x_obj = TRUE iff we have not evaluated the objective function at this point yet
         // pass new_x_cons = TRUE iff we have not evaluated the constraint functions at this point yet
         new_x_obj = last_f_eval_x < current_x;
         new_x_cons = last_g_eval_x < current_x;
      }
      // evaluating Hessians with new_x will also evaluate the functions itself
      last_f_eval_x = current_x;
      last_g_eval_x = current_x;

      if( SCIPnlpiOracleEvalHessianLag(scip, nlpiproblem->oracle, x, new_x_obj, new_x_cons, obj_factor, lambda, values) != SCIP_OKAY )
         return false;
   }

   return true;
}

/** Method called by the solver at each iteration.
 * 
 * Checks whether SCIP solve is interrupted, objlimit is reached, or fastfail is triggered.
 * Sets solution and termination status accordingly.
 */   /*lint -e{715}*/
bool ScipNLP::intermediate_callback(
   AlgorithmMode      mode,               /**< current mode of algorithm */
   Index              iter,               /**< current iteration number */
   Number             obj_value,          /**< current objective value */
   Number             inf_pr,             /**< current primal infeasibility */
   Number             inf_du,             /**< current dual infeasibility */
   Number             mu,                 /**< current barrier parameter */
   Number             d_norm,             /**< current gradient norm */
   Number             regularization_size,/**< current size of regularization */
   Number             alpha_du,           /**< current dual alpha */
   Number             alpha_pr,           /**< current primal alpha */
   Index              ls_trials,          /**< current number of linesearch trials */
   const IpoptData*   ip_data,            /**< pointer to Ipopt Data */
   IpoptCalculatedQuantities* ip_cq       /**< pointer to current calculated quantities */
   )
{  /*lint --e{715}*/
   if( SCIPisSolveInterrupted(scip) )
   {
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_INTERRUPT;
      return false;
   }

   /* feasible point with objective value below lower objective limit -> stop */
   if( obj_value <= param.lobjlimit && inf_pr <= param.feastol )
   {
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_FEASIBLE;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_LOBJLIMIT;
      return false;
   }

   /* do convergence test if fastfail is enabled */
   if( param.fastfail >= 2 )
   {
      int i;

      if( iter == 0 )
      {
         conv_lastrestoiter = -1;
      }
      else if( mode == RestorationPhaseMode )
      {
         conv_lastrestoiter = iter;
      }
      else if( conv_lastrestoiter == iter-1 )
      {
         /* just switched back from restoration mode, reset dual reduction targets */
         for( i = 0; i < convcheck_nchecks; ++i )
            conv_dutarget[i] = convcheck_minred[i] * inf_du;
      }

      if( iter == convcheck_startiter )
      {
         /* define initial targets and iteration limits */
         for( i = 0; i < convcheck_nchecks; ++i )
         {
            conv_prtarget[i] = convcheck_minred[i] * inf_pr;
            conv_dutarget[i] = convcheck_minred[i] * inf_du;
            conv_iterlim[i] = iter + convcheck_maxiter[i];
         }
      }
      else if( iter > convcheck_startiter )
      {
         /* check if we should stop */
         for( i = 0; i < convcheck_nchecks; ++i )
         {
            if( inf_pr <= conv_prtarget[i] )
            {
               /* sufficient reduction w.r.t. primal infeasibility target
                * reset target w.r.t. current infeasibilities
                */
               conv_prtarget[i] = convcheck_minred[i] * inf_pr;
               conv_dutarget[i] = convcheck_minred[i] * inf_du;
               conv_iterlim[i] = iter + convcheck_maxiter[i];
            }
            else if( iter >= conv_iterlim[i] )
            {
               /* we hit a limit, should we really stop? */
               SCIPdebugMsg(scip, "convcheck %d: inf_pr = %e > target %e; inf_du = %e target %e: ",
                  i, inf_pr, conv_prtarget[i], inf_du, conv_dutarget[i]);
               if( mode == RegularMode && iter <= conv_lastrestoiter + convcheck_startiter )
               {
                  /* if we returned from feasibility restoration recently, we allow some more iterations,
                   * because Ipopt may go for optimality for some iterations, at the costs of infeasibility
                   */
                  SCIPdebugPrintf("continue, because restoration phase only %d iters ago\n", iter - conv_lastrestoiter);
               }
               else if( mode == RegularMode && inf_du <= conv_dutarget[i] && iter < conv_iterlim[i] + convcheck_maxiter[i] )
               {
                  /* if dual reduction is sufficient, we allow for twice the number of iterations to reach primal infeas reduction */
                  SCIPdebugPrintf("continue, because dual infeas. red. sufficient and only %d iters above limit\n", iter - conv_iterlim[i]);
               }
               else
               {
                  SCIPdebugPrintf("abort solve\n");
                  if( inf_pr <= param.feastol )
                     nlpiproblem->solstat  = SCIP_NLPSOLSTAT_FEASIBLE;
                  else
                     nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
                  nlpiproblem->termstat = SCIP_NLPTERMSTAT_OKAY;
                  return false;
               }
            }
         }
      }
   }

   return true;
}

/** This method is called when the algorithm is complete so the TNLP can store/write the solution. */  /*lint -e{715}*/
void ScipNLP::finalize_solution(
   SolverReturn       status,             /**< solve and solution status */ 
   Index              n,                  /**< number of variables */ 
   const Number*      x,                  /**< primal solution values */ 
   const Number*      z_L,                /**< dual values of variable lower bounds */
   const Number*      z_U,                /**< dual values of variable upper bounds */
   Index              m,                  /**< number of constraints */ 
   const Number*      g,                  /**< values of constraints */ 
   const Number*      lambda,             /**< dual values of constraints */ 
   Number             obj_value,          /**< objective function value */ 
   const IpoptData*   data,               /**< pointer to Ipopt Data */ 
   IpoptCalculatedQuantities* cq          /**< pointer to calculated quantities */
   )
{ /*lint --e{715}*/
   assert(nlpiproblem         != NULL);
   assert(nlpiproblem->oracle != NULL);

   assert(n == SCIPnlpiOracleGetNVars(nlpiproblem->oracle));
   assert(m == SCIPnlpiOracleGetNConstraints(nlpiproblem->oracle));

   bool check_feasibility = false; // whether we should check x for feasibility, if not NULL
   switch( status )
   {
   case SUCCESS:
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_LOCOPT;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_OKAY;
      assert(x != NULL);
      break;

   case STOP_AT_ACCEPTABLE_POINT:
      /* if stop at acceptable point, then dual infeasibility can be arbitrary large, so claim only feasibility */
   case FEASIBLE_POINT_FOUND:
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_FEASIBLE;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_OKAY;
      assert(x != NULL);
      break;

   case MAXITER_EXCEEDED:
      check_feasibility = true;
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_ITERLIMIT;
      break;

   case CPUTIME_EXCEEDED:
      check_feasibility = true;
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_TIMELIMIT;
      break;

   case STOP_AT_TINY_STEP:
   case RESTORATION_FAILURE:
   case ERROR_IN_STEP_COMPUTATION:
      check_feasibility = true;
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_NUMERICERROR;
      break;

   case LOCAL_INFEASIBILITY:
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_LOCINFEASIBLE;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_OKAY;
      break;

   case DIVERGING_ITERATES:
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNBOUNDED;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_OKAY;
      break;

   case INVALID_NUMBER_DETECTED:
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_EVALERROR;
      break;

   case USER_REQUESTED_STOP:
      // status codes already set in intermediate_callback
      break;

   case TOO_FEW_DEGREES_OF_FREEDOM:
   case INTERNAL_ERROR:
   case INVALID_OPTION:
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_OTHER;
      break;

   case OUT_OF_MEMORY:
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_OUTOFMEMORY;
      break;

   default:
      SCIPerrorMessage("Ipopt returned with unknown solution status %d\n", status);
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_OTHER;
      break;
   }

   assert(x != NULL);
   assert(lambda != NULL);
   assert(z_L != NULL);
   assert(z_U != NULL);

   assert(nlpiproblem->solprimals != NULL);

   if( nlpiproblem->soldualcons == NULL )
   {
      (void) SCIPallocBlockMemoryArray(scip, &nlpiproblem->soldualcons, m);
   }
   if( nlpiproblem->soldualvarlb == NULL )
   {
      (void) SCIPallocBlockMemoryArray(scip, &nlpiproblem->soldualvarlb, n);
   }
   if( nlpiproblem->soldualvarub == NULL )
   {
      (void) SCIPallocBlockMemoryArray(scip, &nlpiproblem->soldualvarub, n);
   }
   if( nlpiproblem->soldualcons == NULL || nlpiproblem->soldualvarlb == NULL || nlpiproblem->soldualvarub == NULL )
   {
      nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
      nlpiproblem->termstat = SCIP_NLPTERMSTAT_OUTOFMEMORY;
      return;
   }

   BMScopyMemoryArray(nlpiproblem->solprimals, x, n);
   BMScopyMemoryArray(nlpiproblem->soldualcons, lambda, m);
   BMScopyMemoryArray(nlpiproblem->soldualvarlb, z_L, n);
   BMScopyMemoryArray(nlpiproblem->soldualvarub, z_U, n);
   nlpiproblem->solobjval = obj_value;
   nlpiproblem->solprimalvalid = true;
   nlpiproblem->solprimalgiven = false;
   nlpiproblem->soldualvalid = true;
   nlpiproblem->soldualgiven = false;

   if( check_feasibility && cq != NULL )
   {
      if( cq->unscaled_curr_nlp_constraint_violation(Ipopt::NORM_MAX) <= param.feastol )
         nlpiproblem->solstat  = SCIP_NLPSOLSTAT_FEASIBLE;
      else if( nlpiproblem->solstat != SCIP_NLPSOLSTAT_LOCINFEASIBLE )
         nlpiproblem->solstat  = SCIP_NLPSOLSTAT_UNKNOWN;
   }

   if( nlpiproblem->solstat == SCIP_NLPSOLSTAT_LOCINFEASIBLE )
   {
      assert(lambda != NULL);
      SCIP_Real tol;
      (void) nlpiproblem->ipopt->Options()->GetNumericValue("tol", tol, "");

      // Jakobs paper ZR_20-20 says we should have lambda*g(x) + mu*h(x) > 0
      //   if the NLP is min f(x) s.t. g(x) <= 0, h(x) = 0
      // we check this here and change solution status to unknown if the test fails
      bool infreasonable = true;
      SCIP_Real infproof = 0.0;
      for( int i = 0; i < m && infreasonable; ++i )
      {
         if( fabs(lambda[i]) < tol )
            continue;
         SCIP_Real side;
         if( lambda[i] < 0.0 )
         {
            // lhs <= g(x) should be active
            // in the NLP above, this should be lhs - g(x) <= 0 with negated dual
            // so this contributes -lambda*(lhs-g(x)) = lambda*(g(x)-side)
            side = SCIPnlpiOracleGetConstraintLhs(nlpiproblem->oracle, i);
            if( SCIPisInfinity(scip, -side) )
            {
               SCIPdebugMessage("inconsistent dual, lambda = %g, but lhs = %g\n", lambda[i], side);
               infreasonable = false;
            }
         }
         else
         {
            // g(x) <= rhs should be active
            // in the NLP above, this should be g(x) - rhs <= 0
            // so this contributes lambda*(g(x)-rhs)
            side = SCIPnlpiOracleGetConstraintRhs(nlpiproblem->oracle, i);
            if( SCIPisInfinity(scip, side) )
            {
               SCIPdebugMessage("inconsistent dual, lambda = %g, but rhs = %g\n", lambda[i], side);
               infreasonable = false;
            }
         }

         // g(x) <= 0
         infproof += lambda[i] * (g[i] - side);
         // SCIPdebugMessage("cons %d lambda %g, slack %g\n", i, lambda[i], g[i] - side);
      }
      if( infreasonable )
      {
         SCIPdebugMessage("infproof = %g should be positive to be valid\n", infproof);
         if( infproof <= 0.0 )
            infreasonable = false;
      }

      if( !infreasonable )
      {
         // change status to say we don't know
         nlpiproblem->solstat = SCIP_NLPSOLSTAT_UNKNOWN;
      }
   }
}

/** Calls Lapacks Dsyev routine to compute eigenvalues and eigenvectors of a dense matrix.
 *
 *  It's here, because we use Ipopt's interface to Lapack.
 */
SCIP_RETCODE LapackDsyev(
   SCIP_Bool             computeeigenvectors,/**< should also eigenvectors should be computed ? */
   int                   N,                  /**< dimension */
   SCIP_Real*            a,                  /**< matrix data on input (size N*N); eigenvectors on output if computeeigenvectors == TRUE */
   SCIP_Real*            w                   /**< buffer to store eigenvalues (size N) */
   )
{
   int info;

#if IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR < 14
   IpLapackDsyev((bool)computeeigenvectors, N, a, N, w, info);
#else
   IpLapackSyev((bool)computeeigenvectors, N, a, N, w, info);
#endif

   if( info != 0 )
   {
      SCIPerrorMessage("There was an error when calling DSYEV. INFO = %d\n", info);
      return SCIP_ERROR;
   }

   return SCIP_OKAY;
}

/** solves a linear problem of the form Ax = b for a regular matrix 3*3 A */
static
SCIP_RETCODE SCIPsolveLinearProb3(
   SCIP_Real*            A,                  /**< matrix data on input (size 3*3); filled column-wise */
   SCIP_Real*            b,                  /**< right hand side vector (size 3) */
   SCIP_Real*            x,                  /**< buffer to store solution (size 3) */
   SCIP_Bool*            success             /**< pointer to store if the solving routine was successful */
   )
{
   SCIP_Real Acopy[9];
   SCIP_Real bcopy[3];
   int pivotcopy[3];
   const int N = 3;
   int info;

   assert(A != NULL);
   assert(b != NULL);
   assert(x != NULL);
   assert(success != NULL);

   BMScopyMemoryArray(Acopy, A, N*N);
   BMScopyMemoryArray(bcopy, b, N);

   /* compute the LU factorization */
#if IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR < 14
   IpLapackDgetrf(N, Acopy, pivotcopy, N, info);
#else
   IpLapackGetrf(N, Acopy, pivotcopy, N, info);
#endif

   if( info != 0 )
   {
      SCIPdebugMessage("There was an error when calling Dgetrf. INFO = %d\n", info);
      *success = FALSE;
   }
   else
   {
      *success = TRUE;

      /* solve linear problem */
#if IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR < 14
      IpLapackDgetrs(N, 1, Acopy, N, pivotcopy, bcopy, N);
#else
      IpLapackGetrs(N, 1, Acopy, N, pivotcopy, bcopy, N);
#endif

      /* copy the solution */
      BMScopyMemoryArray(x, bcopy, N);
   }

   return SCIP_OKAY;
}

/** solves a linear problem of the form Ax = b for a regular matrix A
 *
 *  Calls Lapacks IpLapackDgetrf routine to calculate a LU factorization and uses this factorization to solve
 *  the linear problem Ax = b.
 *  It's here, because Ipopt is linked against Lapack.
 */
SCIP_RETCODE SCIPsolveLinearProb(
   int                   N,                  /**< dimension */
   SCIP_Real*            A,                  /**< matrix data on input (size N*N); filled column-wise */
   SCIP_Real*            b,                  /**< right hand side vector (size N) */
   SCIP_Real*            x,                  /**< buffer to store solution (size N) */
   SCIP_Bool*            success             /**< pointer to store if the solving routine was successful */
   )
{
   SCIP_Real* Acopy;
   SCIP_Real* bcopy;
   int* pivotcopy;
   int info;

   assert(N > 0);
   assert(A != NULL);
   assert(b != NULL);
   assert(x != NULL);
   assert(success != NULL);

   /* call SCIPsolveLinearProb3() for performance reasons */
   if( N == 3 )
   {
      SCIP_CALL( SCIPsolveLinearProb3(A, b, x, success) );
      return SCIP_OKAY;
   }

   Acopy = NULL;
   bcopy = NULL;
   pivotcopy = NULL;

   SCIP_ALLOC( BMSduplicateMemoryArray(&Acopy, A, N*N) );
   SCIP_ALLOC( BMSduplicateMemoryArray(&bcopy, b, N) );
   SCIP_ALLOC( BMSallocMemoryArray(&pivotcopy, N) );

   /* compute the LU factorization */
#if IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR < 14
   IpLapackDgetrf(N, Acopy, pivotcopy, N, info);
#else
   IpLapackGetrf(N, Acopy, pivotcopy, N, info);
#endif

   if( info != 0 )
   {
      SCIPdebugMessage("There was an error when calling Dgetrf. INFO = %d\n", info);
      *success = FALSE;
   }
   else
   {
      *success = TRUE;

      /* solve linear problem */
#if IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR < 14
      IpLapackDgetrs(N, 1, Acopy, N, pivotcopy, bcopy, N);
#else
      IpLapackGetrs(N, 1, Acopy, N, pivotcopy, bcopy, N);
#endif

      /* copy the solution */
      BMScopyMemoryArray(x, bcopy, N);
   }

   BMSfreeMemoryArray(&pivotcopy);
   BMSfreeMemoryArray(&bcopy);
   BMSfreeMemoryArray(&Acopy);

   return SCIP_OKAY;
}

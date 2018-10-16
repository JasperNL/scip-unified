/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2016 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   cons_expr.c
 * @brief  constraint handler for expression constraints (in particular, nonlinear constraints)
 * @author Stefan Vigerske
 * @author Benjamin Mueller
 * @author Felipe Serrano
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>
#include <ctype.h>

#include "scip/cons_expr.h"
#include "scip/cons_linear.h"
#include "scip/struct_cons_expr.h"
#include "scip/cons_expr_var.h"
#include "scip/cons_expr_value.h"
#include "scip/cons_expr_sum.h"
#include "scip/cons_expr_product.h"
#include "scip/cons_expr_exp.h"
#include "scip/cons_expr_log.h"
#include "scip/cons_expr_abs.h"
#include "scip/cons_expr_pow.h"
#include "scip/cons_expr_entropy.h"
#include "scip/cons_expr_sin.h"
#include "scip/cons_expr_cos.h"
#include "scip/cons_expr_nlhdlr_bilinear.h"
#include "scip/cons_expr_nlhdlr_convex.h"
#include "scip/cons_expr_nlhdlr_default.h"
#include "scip/cons_expr_nlhdlr_quadratic.h"
#include "scip/cons_expr_iterator.h"
#include "scip/heur_subnlp.h"
#include "scip/heur_trysol.h"
#include "scip/debug.h"

/* fundamental constraint handler properties */
#define CONSHDLR_NAME          "expr"
#define CONSHDLR_DESC          "constraint handler for expressions"
#define CONSHDLR_ENFOPRIORITY       -60 /**< priority of the constraint handler for constraint enforcing */
#define CONSHDLR_CHECKPRIORITY -4000010 /**< priority of the constraint handler for checking feasibility */
#define CONSHDLR_EAGERFREQ          100 /**< frequency for using all instead of only the useful constraints in separation,
                                         *   propagation and enforcement, -1 for no eager evaluations, 0 for first only */
#define CONSHDLR_NEEDSCONS         TRUE /**< should the constraint handler be skipped, if no constraints are available? */

/* optional constraint handler properties */
#define CONSHDLR_SEPAPRIORITY        10 /**< priority of the constraint handler for separation */
#define CONSHDLR_SEPAFREQ             1 /**< frequency for separating cuts; zero means to separate only in the root node */
#define CONSHDLR_DELAYSEPA        FALSE /**< should separation method be delayed, if other separators found cuts? */

#define CONSHDLR_PROPFREQ             1 /**< frequency for propagating domains; zero means only preprocessing propagation */
#define CONSHDLR_DELAYPROP        FALSE /**< should propagation method be delayed, if other propagators found reductions? */
#define CONSHDLR_PROP_TIMING     SCIP_PROPTIMING_BEFORELP /**< propagation timing mask of the constraint handler*/

#define CONSHDLR_PRESOLTIMING    SCIP_PRESOLTIMING_ALWAYS /**< presolving timing of the constraint handler (fast, medium, or exhaustive) */
#define CONSHDLR_MAXPREROUNDS        -1 /**< maximal number of presolving rounds the constraint handler participates in (-1: no limit) */

/* properties of the expression constraint handler statistics table */
#define TABLE_NAME_EXPR                          "expression"
#define TABLE_DESC_EXPR                          "expression constraint handler statistics"
#define TABLE_POSITION_EXPR                      12500                  /**< the position of the statistics table */
#define TABLE_EARLIEST_STAGE_EXPR                SCIP_STAGE_TRANSFORMED /**< output of the statistics table is only printed from this stage onwards */

/* enable nonlinear constraint upgrading */
#include "scip/cons_nonlinear.h"
#define NONLINCONSUPGD_PRIORITY   600000 /**< priority of the constraint handler for upgrading of nonlinear constraints */

/* enable quadratic constraint upgrading */
#include "scip/cons_quadratic.h"
#define QUADCONSUPGD_PRIORITY     600000 /**< priority of the constraint handler for upgrading of quadratic constraints */

/** ensures that a block memory array has at least a given size
 *
 *  if cursize is 0, then *array1 can be NULL
 */
#define ENSUREBLOCKMEMORYARRAYSIZE(scip, array1, cursize, minsize)      \
   do {                                                                 \
      int __newsize;                                                    \
      assert((scip)  != NULL);                                          \
      if( (cursize) >= (minsize) )                                      \
         break;                                                         \
      __newsize = SCIPcalcMemGrowSize(scip, minsize);                   \
      assert(__newsize >= (minsize));                                   \
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(array1), cursize, __newsize) ); \
      (cursize) = __newsize;                                            \
   } while( FALSE )

/** translate from one value of infinity to another
 *
 *  if val is >= infty1, then give infty2, else give val
 */
#define infty2infty(infty1, infty2, val) ((val) >= (infty1) ? (infty2) : (val))


/*
 * Data structures
 */

/** eventdata for variable bound change events in constraints */
typedef struct
{
   SCIP_CONS*            cons;               /**< constraint */
   SCIP_CONSEXPR_EXPR*   varexpr;            /**< variable expression */
   int                   filterpos;          /**< position of eventdata in SCIP's event filter */
} SCIP_VAREVENTDATA;

/** expression constraint update method */
struct SCIP_ExprConsUpgrade
{
   SCIP_DECL_EXPRCONSUPGD((*exprconsupgd));  /**< method to call for upgrading expression constraint */
   int                   priority;           /**< priority of upgrading method */
   SCIP_Bool             active;             /**< is upgrading enabled */
};
typedef struct SCIP_ExprConsUpgrade SCIP_EXPRCONSUPGRADE;

/** constraint data for expr constraints */
struct SCIP_ConsData
{
   SCIP_CONSEXPR_EXPR*   expr;               /**< expression that represents this constraint */
   SCIP_Real             lhs;                /**< left-hand side */
   SCIP_Real             rhs;                /**< right-hand side */

   SCIP_CONSEXPR_EXPR**  varexprs;           /**< array containing all variable expressions */
   int                   nvarexprs;          /**< total number of variable expressions */
   SCIP_VAREVENTDATA**   vareventdata;       /**< array containing eventdata for bound change of variables */

   SCIP_Real             lhsviol;            /**< violation of left-hand side by current solution (used temporarily inside constraint handler) */
   SCIP_Real             rhsviol;            /**< violation of right-hand side by current solution (used temporarily inside constraint handler) */

   unsigned int          ispropagated:1;     /**< did we propagate the current bounds already? */
   unsigned int          issimplified:1;     /**< did we simplify the expression tree already? */

   SCIP_NLROW*           nlrow;              /**< a nonlinear row representation of this constraint */

   int                   nlockspos;          /**< number of positive locks */
   int                   nlocksneg;          /**< number of negative locks */

   /* repair infeasible solutions */
   SCIP_VAR*             linvardecr;         /**< variable that may be decreased without making any other constraint infeasible, or NULL if none */
   SCIP_VAR*             linvarincr;         /**< variable that may be increased without making any other constraint infeasible, or NULL if none */
   SCIP_Real             linvardecrcoef;     /**< linear coefficient of linvardecr */
   SCIP_Real             linvarincrcoef;     /**< linear coefficient of linvarincr */
};

/** constraint handler data */
struct SCIP_ConshdlrData
{
   /* expression handler */
   SCIP_CONSEXPR_EXPRHDLR** exprhdlrs;       /**< expression handlers */
   int                      nexprhdlrs;      /**< number of expression handlers */
   int                      exprhdlrssize;   /**< size of exprhdlrs array */

   SCIP_CONSEXPR_EXPRHDLR*  exprvarhdlr;     /**< variable expression handler */
   SCIP_CONSEXPR_EXPRHDLR*  exprvalhdlr;     /**< value expression handler */
   SCIP_CONSEXPR_EXPRHDLR*  exprsumhdlr;     /**< summation expression handler */
   SCIP_CONSEXPR_EXPRHDLR*  exprprodhdlr;    /**< product expression handler */
   SCIP_CONSEXPR_EXPRHDLR*  exprpowhdlr;     /**< power expression handler */

   /* nonlinear handler */
   SCIP_CONSEXPR_NLHDLR**   nlhdlrs;         /**< nonlinear handlers */
   int                      nnlhdlrs;        /**< number of nonlinear handlers */
   int                      nlhdlrssize;     /**< size of nlhdlrs array */

   /* constraint upgrades */
   SCIP_EXPRCONSUPGRADE**   exprconsupgrades;     /**< nonlinear constraint upgrade methods for specializing expression constraints */
   int                      exprconsupgradessize; /**< size of exprconsupgrades array */
   int                      nexprconsupgrades;    /**< number of expression constraint upgrade methods */

   /* other plugins */
   SCIP_EVENTHDLR*          eventhdlr;       /**< handler for variable bound change events */
   SCIP_HEUR*               subnlpheur;      /**< a pointer to the subnlp heuristic, if available */
   SCIP_HEUR*               trysolheur;      /**< a pointer to the trysol heuristic, if available */

   /* expression iterator */
   int                      nactiveiter;     /**< number of currently active iterators */
   unsigned int             lastvisitedtag;  /**< last visited tag used by iterators */

   /* tags and counters */
   int                      auxvarid;        /**< unique id for the next auxiliary variable */

   unsigned int             lastsoltag;      /**< last solution tag used to evaluate current solution */
   unsigned int             lastintevaltag;  /**< last interval evaluation tag used */
   unsigned int             lastdifftag;     /**< last tag used for computing gradients */
   unsigned int             lastbrscoretag;  /**< last branching score tag used */

   SCIP_Longint             lastenfolpnodenum; /**< number of node for which enforcement has been called last */
   SCIP_Longint             lastenfopsnodenum; /**< number of node for which enforcement has been called last */
   SCIP_Longint             lastpropnodenum; /**< number node for which propagation has been called last */

   /* parameters */
   int                      maxproprounds;   /**< limit on number of propagation rounds for a set of constraints within one round of SCIP propagation */
   char                     varboundrelax;   /**< strategy on how to relax variable bounds during bound tightening */
   SCIP_Real                varboundrelaxamount; /**< by how much to relax variable bounds during bound tightening */
   SCIP_Real                conssiderelaxamount; /**< by how much to relax constraint sides during bound tightening */

   /* statistics */
   SCIP_Longint             ndesperatebranch;/**< number of times we branched on some variable because normal enforcement was not successful */
   SCIP_Longint             ndesperatecutoff;/**< number of times we cut off a node in enforcement because no branching candidate could be found */
   SCIP_Longint             nforcelp;        /**< number of times we forced solving the LP when enforcing a pseudo solution */
   SCIP_CLOCK*              canonicalizetime;/**< time spend for canonicalization */
};

/** variable mapping data passed on during copying expressions when copying SCIP instances */
typedef struct
{
   SCIP_HASHMAP*           varmap;           /**< SCIP_HASHMAP mapping variables of the source SCIP to corresponding variables of the target SCIP */
   SCIP_HASHMAP*           consmap;          /**< SCIP_HASHMAP mapping constraints of the source SCIP to corresponding constraints of the target SCIP */
   SCIP_Bool               global;           /**< should a global or a local copy be created */
   SCIP_Bool               valid;            /**< indicates whether every variable copy was valid */
} COPY_MAPVAR_DATA;

/** printing to dot file data */
struct SCIP_ConsExpr_PrintDotData
{
   FILE*                   file;             /**< file to print to */
   SCIP_CONSEXPR_ITERATOR* iterator;         /**< iterator to use */
   SCIP_Bool               closefile;        /**< whether file need to be closed when finished printing */
   SCIP_HASHMAP*           leaveexprs;       /**< hashmap storing leave (no children) expressions */
   SCIP_CONSEXPR_PRINTDOT_WHAT whattoprint;  /**< flags that indicate what to print for each expression */
};

/*
 * Local methods
 */

/** creates an expression */
static
SCIP_RETCODE createExpr(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR**    expr,             /**< pointer where to store expression */
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr,         /**< expression handler */
   SCIP_CONSEXPR_EXPRDATA* exprdata,         /**< expression data (expression assumes ownership) */
   int                     nchildren,        /**< number of children */
   SCIP_CONSEXPR_EXPR**    children          /**< children (can be NULL if nchildren is 0) */
   )
{
   int c;

   assert(expr != NULL);
   assert(exprhdlr != NULL);
   assert(children != NULL || nchildren == 0);

   SCIP_CALL( SCIPallocClearBlockMemory(scip, expr) );

   (*expr)->exprhdlr = exprhdlr;
   (*expr)->exprdata = exprdata;
   (*expr)->curvature = SCIP_EXPRCURV_UNKNOWN;

   /* initialize an empty interval for interval evaluation */
   SCIPintervalSetEntire(SCIP_INTERVAL_INFINITY, &(*expr)->interval);

   if( nchildren > 0 )
   {
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &(*expr)->children, children, nchildren) );
      (*expr)->nchildren = nchildren;
      (*expr)->childrensize = nchildren;

      for( c = 0; c < nchildren; ++c )
         SCIPcaptureConsExprExpr((*expr)->children[c]);
   }

   SCIPcaptureConsExprExpr(*expr);

   return SCIP_OKAY;
}

/** frees an expression */
static
SCIP_RETCODE freeExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR**  expr                /**< pointer to free the expression */
   )
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(*expr != NULL);
   assert((*expr)->nuses == 1);

   /* free children array, if any */
   SCIPfreeBlockMemoryArrayNull(scip, &(*expr)->children, (*expr)->childrensize);

   /* expression should not be locked anymore */
   assert((*expr)->nlockspos == 0);
   assert((*expr)->nlocksneg == 0);

   SCIPfreeBlockMemory(scip, expr);
   assert(*expr == NULL);

   return SCIP_OKAY;
}

/** frees auxiliary variables of expression, if any */
static
SCIP_RETCODE freeAuxVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*   expr                /**< expression which auxvar to free, if any */
   )
{
   assert(scip != NULL);
   assert(expr != NULL);

   if( expr->auxvar == NULL )
      return SCIP_OKAY;

   SCIPdebugMsg(scip, "remove auxiliary variable %s for expression %p\n", SCIPvarGetName(expr->auxvar), (void*)expr);

   /* remove variable locks if variable is not used by any other plug-in which can be done by checking whether
    * SCIPvarGetNUses() returns 2 (1 for the core; and one for cons_expr); note that SCIP does not enforce to have 0
    * locks when freeing a variable
    */
   assert(SCIPvarGetNUses(expr->auxvar) >= 2);
   if( SCIPvarGetNUses(expr->auxvar) == 2 )
   {
      SCIP_CALL( SCIPaddVarLocks(scip, expr->auxvar, -1, -1) );
   }

   /* release auxiliary variable */
   SCIP_CALL( SCIPreleaseVar(scip, &expr->auxvar) );
   assert(expr->auxvar == NULL);

   return SCIP_OKAY;
}

/** frees data used for enforcement, that is, nonlinear handlers and auxiliary variables */
static
SCIP_RETCODE freeEnfoData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression whose enforcement data will be released */
   SCIP_Bool             freeauxvar          /**< whether aux var should be released */
   )
{
   int e;

   /* free auxiliary variable */
   if( freeauxvar )
   {
      SCIP_CALL( freeAuxVar(scip, expr) );
      assert(expr->auxvar == NULL);
   }

   /* free data stored by nonlinear handlers */
   for( e = 0; e < expr->nenfos; ++e )
   {
      SCIP_CONSEXPR_NLHDLR* nlhdlr;

      assert(expr->enfos[e] != NULL);

      nlhdlr = expr->enfos[e]->nlhdlr;
      assert(nlhdlr != NULL);

      if( expr->enfos[e]->issepainit )
      {
         /* call the separation deinitialization callback of the nonlinear handler */
         SCIP_CALL( SCIPexitsepaConsExprNlhdlr(scip, nlhdlr, expr, expr->enfos[e]->nlhdlrexprdata) );
         expr->enfos[e]->issepainit = FALSE;
      }

      /* free nlhdlr exprdata, if there is any and there is a method to free this data */
      if( expr->enfos[e]->nlhdlrexprdata != NULL && nlhdlr->freeexprdata != NULL )
      {
         SCIP_CALL( (*nlhdlr->freeexprdata)(scip, nlhdlr, &expr->enfos[e]->nlhdlrexprdata) );
         assert(expr->enfos[e]->nlhdlrexprdata == NULL);
      }

      /* free enfo data */
      SCIPfreeBlockMemory(scip, &expr->enfos[e]); /*lint !e866 */
   }

   /* free array with enfo data */
   SCIPfreeBlockMemoryArrayNull(scip, &expr->enfos, expr->nenfos);
   expr->nenfos = 0;

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_MAPVAR(transformVar)
{   /*lint --e{715}*/
   assert(sourcevar != NULL);
   assert(targetvar != NULL);
   assert(sourcescip == targetscip);

   /* transform variable (does not capture target variable) */
   SCIP_CALL( SCIPgetTransformedVar(sourcescip, sourcevar, targetvar) );
   assert(*targetvar != NULL);

   /* caller assumes that target variable has been captured */
   SCIP_CALL( SCIPcaptureVar(sourcescip, *targetvar) );

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_MAPVAR(copyVar)
{
   COPY_MAPVAR_DATA* data;
   SCIP_Bool valid;

   assert(sourcevar != NULL);
   assert(targetvar != NULL);
   assert(mapvardata != NULL);

   data = (COPY_MAPVAR_DATA*)mapvardata;

   SCIP_CALL( SCIPgetVarCopy(sourcescip, targetscip, sourcevar, targetvar, data->varmap, data->consmap, data->global, &valid) );
   assert(*targetvar != NULL);

   /* if copy was not valid, store so in mapvar data */
   if( !valid )
      data->valid = FALSE;

   /* caller assumes that target variable has been captured */
   SCIP_CALL( SCIPcaptureVar(targetscip, *targetvar) );

   return SCIP_OKAY;
}

/** copies an expression including subexpressions
 *
 * \note If copying fails due to missing copy callbacks, then *targetexpr will be set to NULL.
 */
static
SCIP_RETCODE copyExpr(
   SCIP*                 sourcescip,         /**< SCIP data structure corresponding to source expression */
   SCIP*                 targetscip,         /**< SCIP data structure where target expression will live */
   SCIP_CONSHDLR*        consexprhdlr,       /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*   sourceexpr,         /**< expression to be copied */
   SCIP_CONSEXPR_EXPR**  targetexpr,         /**< buffer to store pointer to copy of source expression */
   SCIP_DECL_CONSEXPR_MAPVAR((*mapvar)),  /**< variable mapping function, or NULL for identity mapping */
   void*                 mapvardata          /**< data of variable mapping function */
   )
{
   SCIP_CONSHDLR* targetconsexprhdlr = NULL;
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSEXPRITERATOR_USERDATA expriteruserdata;
   SCIP_CONSEXPR_EXPR* expr;

   assert(sourcescip != NULL);
   assert(targetscip != NULL);
   assert(consexprhdlr != NULL);
   assert(sourceexpr != NULL);
   assert(targetexpr != NULL);

   if( targetscip != sourcescip )
      targetconsexprhdlr = SCIPfindConshdlr(targetscip, CONSHDLR_NAME);
   else
      targetconsexprhdlr = consexprhdlr;
   assert(targetconsexprhdlr != NULL);

   SCIP_CALL( SCIPexpriteratorCreate(&it, consexprhdlr, SCIPblkmem(sourcescip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, sourceexpr, SCIP_CONSEXPRITERATOR_DFS, TRUE) );  /*TODO use FALSE, i.e., don't duplicate common subexpr? */
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_ENTEREXPR | SCIP_CONSEXPRITERATOR_VISITEDCHILD);

   expr = sourceexpr;
   while( !SCIPexpriteratorIsEnd(it) )
   {
      switch( SCIPexpriteratorGetStageDFS(it) )
      {
         case SCIP_CONSEXPRITERATOR_ENTEREXPR :
         {
            /* create expr that will hold the copy */
            SCIP_CONSEXPR_EXPR* exprcopy;

            /* if the source is a variable expression create a variable expression directly; otherwise copy the expression data */
            if( SCIPisConsExprExprVar(expr) )
            {
               SCIP_VAR* sourcevar;
               SCIP_VAR* targetvar;

               sourcevar = SCIPgetConsExprExprVarVar(expr);
               assert(sourcevar != NULL);
               targetvar = NULL;

               /* get the corresponding variable in the target SCIP */
               if( mapvar != NULL )
               {
                  SCIP_CALL( mapvar(targetscip, &targetvar, sourcescip, sourcevar, mapvardata) );
                  SCIP_CALL( SCIPcreateConsExprExprVar(targetscip, targetconsexprhdlr, &exprcopy, targetvar) );

                  /* we need to release once since it has been captured by the mapvar() and SCIPcreateConsExprExprVar() call */
                  SCIP_CALL( SCIPreleaseVar(targetscip, &targetvar) );
               }
               else
               {
                  targetvar = sourcevar;
                  SCIP_CALL( SCIPcreateConsExprExprVar(targetscip, targetconsexprhdlr, &exprcopy, targetvar) );
               }
            }
            else
            {
               SCIP_CONSEXPR_EXPRHDLR* targetexprhdlr;
               SCIP_CONSEXPR_EXPRDATA* targetexprdata;

               /* get the exprhdlr of the target scip */
               if( targetscip != sourcescip )
               {
                  assert(targetconsexprhdlr != NULL);

                  targetexprhdlr = SCIPfindConsExprExprHdlr(targetconsexprhdlr,
                     SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)));

                  if( targetexprhdlr == NULL )
                  {
                     /* expression handler not in target scip (probably did not have a copy callback) -> abort */
                     expriteruserdata.ptrval = NULL;
                     SCIPexpriteratorSetCurrentUserData(it, expriteruserdata);

                     expr = SCIPexpriteratorSkipDFS(it);
                     continue;
                  }
               }
               else
               {
                  targetexprhdlr = SCIPgetConsExprExprHdlr(expr);
               }
               assert(targetexprhdlr != NULL);

               /* copy expression data */
               if( expr->exprhdlr->copydata != NULL )
               {
                  SCIP_CALL( expr->exprhdlr->copydata(targetscip, targetexprhdlr, &targetexprdata, sourcescip, expr, mapvar, mapvardata) );
               }
               else if( expr->exprdata != NULL )
               {
                  /* no copy callback for expression data implemented -> abort
                   * (we could also just copy the exprdata pointer, but for now let's say that
                   *  an expression handler should explicitly implement this behavior, if desired)
                   */
                  expriteruserdata.ptrval = NULL;
                  SCIPexpriteratorSetCurrentUserData(it, expriteruserdata);

                  expr = SCIPexpriteratorSkipDFS(it);
                  continue;
               }
               else
               {
                  targetexprdata = NULL;
               }

               /* create in targetexpr an expression of the same type as expr, but without children for now */
               SCIP_CALL( SCIPcreateConsExprExpr(targetscip, &exprcopy, targetexprhdlr, targetexprdata, 0, NULL) );
            }

            /* store targetexpr */
            expriteruserdata.ptrval = exprcopy;
            SCIPexpriteratorSetCurrentUserData(it, expriteruserdata);

            break;
         }

         case SCIP_CONSEXPRITERATOR_VISITEDCHILD :
         {
            /* just visited child so a copy of himself should be available; append it */
            SCIP_CONSEXPR_EXPR* exprcopy;
            SCIP_CONSEXPR_EXPR* childcopy;

            exprcopy = (SCIP_CONSEXPR_EXPR*)SCIPexpriteratorGetCurrentUserData(it).ptrval;

            /* get copy of child */
            childcopy = (SCIP_CONSEXPR_EXPR*)SCIPexpriteratorGetChildUserDataDFS(it).ptrval;
            if( childcopy == NULL )
            {
               /* abort */
               /* release exprcopy (should free also the already copied children) */
               SCIP_CALL( SCIPreleaseConsExprExpr(targetscip, (SCIP_CONSEXPR_EXPR**)&exprcopy) );

               expriteruserdata.ptrval = NULL;
               SCIPexpriteratorSetCurrentUserData(it, expriteruserdata);

               expr = SCIPexpriteratorSkipDFS(it);
               continue;
            }

            /* append child to exprcopy */
            SCIP_CALL( SCIPappendConsExprExpr(targetscip, exprcopy, childcopy) );

            /* release childcopy (still captured by exprcopy) */
            SCIP_CALL( SCIPreleaseConsExprExpr(targetscip, &childcopy) );

            break;
         }

         default:
            /* we should never be called in this stage */
            SCIPABORT();
            break;
      }

      expr = SCIPexpriteratorGetNext(it);
   }

   /* the target expression should be stored in the userdata of the sourceexpr (can be NULL if aborted) */
   *targetexpr = (SCIP_CONSEXPR_EXPR*)SCIPexpriteratorGetExprUserData(it, sourceexpr).ptrval;

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** create and include conshdlr to SCIP and set everything except for expression handlers */
static
SCIP_RETCODE includeConshdlrExprBasic(SCIP* scip);

/** copy expression and nonlinear handlers from sourceconshdlr to (target's) scip consexprhdlr */
static
SCIP_RETCODE copyConshdlrExprExprHdlr(
   SCIP*                 scip,               /**< (target) SCIP data structure */
   SCIP_CONSHDLR*        sourceconshdlr,     /**< source constraint expression handler */
   SCIP_Bool*            valid               /**< was the copying process valid? */
   )
{
   int                i;
   SCIP_CONSHDLR*     conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSHDLRDATA* sourceconshdlrdata;

   assert(strcmp(SCIPconshdlrGetName(sourceconshdlr), CONSHDLR_NAME) == 0);

   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);
   assert(conshdlr != sourceconshdlr);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   sourceconshdlrdata = SCIPconshdlrGetData(sourceconshdlr);
   assert(sourceconshdlrdata != NULL);

   /* copy expression handlers */
   *valid = TRUE;
   for( i = 0; i < sourceconshdlrdata->nexprhdlrs; i++ )
   {
      SCIP_Bool localvalid;
      SCIP_CONSEXPR_EXPRHDLR* sourceexprhdlr;

      sourceexprhdlr = sourceconshdlrdata->exprhdlrs[i];

      if( sourceexprhdlr->copyhdlr != NULL )
      {
         SCIP_CALL( sourceexprhdlr->copyhdlr(scip, conshdlr, sourceconshdlr, sourceexprhdlr, &localvalid) );
         *valid &= localvalid;
      }
      else
      {
         *valid = FALSE;
      }
   }

   /* set pointer to important expression handlers in conshdlr of target SCIP */
   conshdlrdata->exprvarhdlr = SCIPfindConsExprExprHdlr(conshdlr, "var");
   conshdlrdata->exprvalhdlr = SCIPfindConsExprExprHdlr(conshdlr, "val");
   conshdlrdata->exprsumhdlr = SCIPfindConsExprExprHdlr(conshdlr, "sum");
   conshdlrdata->exprprodhdlr = SCIPfindConsExprExprHdlr(conshdlr, "prod");
   conshdlrdata->exprpowhdlr = SCIPfindConsExprExprHdlr(conshdlr, "pow");

   /* copy nonlinear handlers */
   for( i = 0; i < sourceconshdlrdata->nnlhdlrs; ++i )
   {
      SCIP_CONSEXPR_NLHDLR* sourcenlhdlr;

      /* TODO for now just don't copy disabled nlhdlr, we clean way would probably to copy them and disable then */
      sourcenlhdlr = sourceconshdlrdata->nlhdlrs[i];
      if( sourcenlhdlr->copyhdlr != NULL && sourcenlhdlr->enabled )
      {
         SCIP_CALL( sourcenlhdlr->copyhdlr(scip, conshdlr, sourceconshdlr, sourcenlhdlr) );
      }
   }

   return SCIP_OKAY;
}

/** tries to automatically convert an expression constraint into a more specific and more specialized constraint */
static
SCIP_RETCODE presolveUpgrade(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler data structure */
   SCIP_CONS*            cons,               /**< source constraint to try to convert */
   SCIP_Bool*            upgraded,           /**< buffer to store whether constraint was upgraded */
   int*                  nupgdconss,         /**< buffer to increase if constraint was upgraded */
   int*                  naddconss           /**< buffer to increase with number of additional constraints created during upgrade */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONS** upgdconss;
   int upgdconsssize;
   int nupgdconss_;
   int i;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(cons != NULL);
   assert(!SCIPconsIsModifiable(cons));
   assert(upgraded   != NULL);
   assert(nupgdconss != NULL);
   assert(naddconss  != NULL);

   *upgraded = FALSE;

   nupgdconss_ = 0;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* if there are no upgrade methods, we can stop */
   if( conshdlrdata->nexprconsupgrades == 0 )
      return SCIP_OKAY;

   upgdconsssize = 2;
   SCIP_CALL( SCIPallocBufferArray(scip, &upgdconss, upgdconsssize) );

   /* call the upgrading methods */
   SCIPdebugMsg(scip, "upgrading expression constraint <%s> (up to %d upgrade methods): ",
      SCIPconsGetName(cons), conshdlrdata->nexprconsupgrades);
   SCIPdebugPrintCons(scip, cons, NULL);

   /* try all upgrading methods in priority order in case the upgrading step is enable  */
   for( i = 0; i < conshdlrdata->nexprconsupgrades; ++i )
   {
      if( !conshdlrdata->exprconsupgrades[i]->active )
         continue;

      assert(conshdlrdata->exprconsupgrades[i]->exprconsupgd != NULL);

      SCIP_CALL( conshdlrdata->exprconsupgrades[i]->exprconsupgd(scip, cons, &nupgdconss_, upgdconss, upgdconsssize) );

      while( nupgdconss_ < 0 )
      {
         /* upgrade function requires more memory: resize upgdconss and call again */
         assert(-nupgdconss_ > upgdconsssize);
         upgdconsssize = -nupgdconss_;
         SCIP_CALL( SCIPreallocBufferArray(scip, &upgdconss, -nupgdconss_) );

         SCIP_CALL( conshdlrdata->exprconsupgrades[i]->exprconsupgd(scip, cons, &nupgdconss_, upgdconss, upgdconsssize) );

         assert(nupgdconss_ != 0);
      }

      if( nupgdconss_ > 0 )
      {
         /* got upgrade */
         int j;

         SCIPdebugMsg(scip, " -> upgraded to %d constraints:\n", nupgdconss_);

         /* add the upgraded constraints to the problem and forget them */
         for( j = 0; j < nupgdconss_; ++j )
         {
            SCIPdebugMsgPrint(scip, "\t");
            SCIPdebugPrintCons(scip, upgdconss[j], NULL);

            SCIP_CALL( SCIPaddCons(scip, upgdconss[j]) );      /*lint !e613*/
            SCIP_CALL( SCIPreleaseCons(scip, &upgdconss[j]) ); /*lint !e613*/
         }

         /* count the first upgrade constraint as constraint upgrade and the remaining ones as added constraints */
         *nupgdconss += 1;
         *naddconss += nupgdconss_ - 1;
         *upgraded = TRUE;

         /* delete upgraded constraint */
         SCIPdebugMsg(scip, "delete constraint <%s> after upgrade\n", SCIPconsGetName(cons));
         SCIP_CALL( SCIPdelCons(scip, cons) );

         break;
      }
   }

   SCIPfreeBufferArray(scip, &upgdconss);

   return SCIP_OKAY;
}

/** interval evaluation of variables as used in bound tightening
 *
 * Returns slightly relaxed local variable bounds of a variable as interval.
 * Does not relax beyond integer values, thus does not relax bounds on integer variables at all.
 */
static
SCIP_DECL_CONSEXPR_INTEVALVAR(intEvalVarBoundTightening)
{
   SCIP_INTERVAL interval;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_Real lb;
   SCIP_Real ub;
   SCIP_Real bnd;

   assert(scip != NULL);
   assert(var != NULL);

   conshdlrdata = (SCIP_CONSHDLRDATA*)intevalvardata;
   assert(conshdlrdata != NULL);

   lb = SCIPvarGetLbLocal(var);
   ub = SCIPvarGetUbLocal(var);
   assert(lb <= ub);  /* can SCIP ensure by now that variable bounds are not contradicting? */

   /* implicit integer variables may have non-integer bounds, apparently (run space25a) */
   if( SCIPvarGetType(var) == SCIP_VARTYPE_IMPLINT )
   {
      lb = EPSROUND(lb, 0.0); /*lint !e835*/
      ub = EPSROUND(ub, 0.0); /*lint !e835*/
   }

   /* integer variables should always have integral bounds in SCIP */
   assert(EPSFRAC(lb, 0.0) == 0.0 || !SCIPvarIsIntegral(var));  /*lint !e835*/
   assert(EPSFRAC(ub, 0.0) == 0.0 || !SCIPvarIsIntegral(var));  /*lint !e835*/

   switch( conshdlrdata->varboundrelax )
   {
      case 'n' : /* no relaxation */
         break;

      case 'a' : /* relax by absolute value */
      {
         /* do not look at integer variables, they already have integral bounds, so wouldn't be relaxed */
         if( SCIPvarIsIntegral(var) )
            break;

         if( !SCIPisInfinity(scip, -lb) )
         {
            /* reduce lb by epsilon, or to the next integer value, which ever is larger */
            bnd = floor(lb);
            lb = MAX(bnd, lb - conshdlrdata->varboundrelaxamount);
         }

         if( !SCIPisInfinity(scip, ub) )
         {
            /* increase ub by epsilon, or to the next integer value, which ever is smaller */
            bnd = ceil(ub);
            ub = MIN(bnd, ub + conshdlrdata->varboundrelaxamount);
         }

         break;
      }

      case 'r' : /* relax by relative value */
      {
         /* do not look at integer variables, they already have integral bounds, so wouldn't be relaxed */
         if( SCIPvarIsIntegral(var) )
            break;

         if( !SCIPisInfinity(scip, -lb) )
         {
            /* reduce lb by epsilon*|lb|, or to the next integer value, which ever is larger */
            bnd = floor(lb);
            lb = MAX(bnd, lb - REALABS(lb) * conshdlrdata->varboundrelaxamount);  /*lint !e666*/
         }

         if( !SCIPisInfinity(scip, ub) )
         {
            /* increase ub by epsilon*|ub|, or to the next integer value, which ever is smaller */
            bnd = ceil(ub);
            ub = MIN(bnd, ub + REALABS(ub) * conshdlrdata->varboundrelaxamount);  /*lint !e666*/
         }

         break;
      }

      default :
      {
         SCIPerrorMessage("Unsupported value '%c' for varboundrelax option.\n");
         SCIPABORT();
         break;
      }
   }

   /* convert SCIPinfinity() to SCIP_INTERVAL_INFINITY */
   lb = -infty2infty(SCIPinfinity(scip), SCIP_INTERVAL_INFINITY, -lb);
   ub =  infty2infty(SCIPinfinity(scip), SCIP_INTERVAL_INFINITY, ub);
   assert(lb <= ub);

   SCIPintervalSetBounds(&interval, lb, ub);

   return interval;
}

/** interval evaluation of variables as used in redundancy check
 *
 * Returns local variable bounds of a variable, relaxed by feastol, as interval.
 */
static
SCIP_DECL_CONSEXPR_INTEVALVAR(intEvalVarRedundancyCheck)
{  /*lint --e{715}*/
   SCIP_INTERVAL interval;
   SCIP_Real lb;
   SCIP_Real ub;

   assert(scip != NULL);
   assert(var != NULL);

   lb = SCIPvarGetLbLocal(var);
   ub = SCIPvarGetUbLocal(var);
   assert(lb <= ub);  /* can SCIP ensure by now that variable bounds are not contradicting? */

   /* TODO maybe we should not relax fixed variables? */

   /* relax variable bounds */
   if( !SCIPisInfinity(scip, -lb) )
      lb -= SCIPfeastol(scip);

   if( !SCIPisInfinity(scip, ub) )
      ub += SCIPfeastol(scip);

   /* convert SCIPinfinity() to SCIP_INTERVAL_INFINITY */
   lb = -infty2infty(SCIPinfinity(scip), SCIP_INTERVAL_INFINITY, -lb);
   ub =  infty2infty(SCIPinfinity(scip), SCIP_INTERVAL_INFINITY,  ub);
   assert(lb <= ub);

   SCIPintervalSetBounds(&interval, lb, ub);

   return interval;
}

/** propagate bounds of the expressions in a given expression tree and tries to tighten the bounds of the auxiliary
 *  variables accordingly
 */
static
SCIP_RETCODE forwardPropExpr(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          consexprhdlr,     /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*     rootexpr,         /**< expression */
   SCIP_Bool               force,            /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool               tightenauxvars,   /**< should the bounds of auxiliary variables be tightened? */
   SCIP_DECL_CONSEXPR_INTEVALVAR((*intevalvar)), /**< function to call to evaluate interval of variable, or NULL */
   void*                   intevalvardata,   /**< data to be passed to intevalvar call */
   unsigned int            boxtag,           /**< tag that uniquely identifies the current variable domains (with its values), or 0 */
   SCIP_Bool*              infeasible,       /**< buffer to store whether the problem is infeasible (NULL if not needed) */
   int*                    ntightenings      /**< buffer to store the number of auxiliary variable tightenings (NULL if not needed) */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSEXPR_EXPR* expr;
   SCIP_Bool aborted;

   assert(scip != NULL);
   assert(rootexpr != NULL);

   if( infeasible != NULL )
      *infeasible = FALSE;
   if( ntightenings != NULL )
      *ntightenings = 0;

   /* if value is up-to-date, then nothing to do */
   if( boxtag != 0 && rootexpr->intevaltag == boxtag && !rootexpr->hastightened )
      return SCIP_OKAY;

   SCIP_CALL( SCIPexpriteratorCreate(&it, consexprhdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, rootexpr, SCIP_CONSEXPRITERATOR_DFS, TRUE) );
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_VISITINGCHILD | SCIP_CONSEXPRITERATOR_LEAVEEXPR);

   for( expr = SCIPexpriteratorGetCurrent(it), aborted = FALSE; !SCIPexpriteratorIsEnd(it) && !aborted;  )
   {
      switch( SCIPexpriteratorGetStageDFS(it) )
      {
         case SCIP_CONSEXPRITERATOR_VISITINGCHILD :
         {
            SCIP_CONSEXPR_EXPR* child;

            child = SCIPexpriteratorGetChildExprDFS(it);
            if( boxtag != 0 && boxtag == child->intevaltag && !expr->hastightened )
            {
               /* skip child if it has been evaluated already and current expr is not tightened */
               if( SCIPintervalIsEmpty(SCIP_INTERVAL_INFINITY, child->interval) )
               {
                  aborted = TRUE;
                  break;
               }

               expr = SCIPexpriteratorSkipDFS(it);
               continue;
            }

            break;
         }

         case SCIP_CONSEXPRITERATOR_LEAVEEXPR :
         {
            SCIP_Bool intersect;
            SCIP_INTERVAL interval;

            /* reset interval of the expression if using boxtag = 0 or we did not visit this expression so
             * far, i.e., expr->intevaltag != boxtag
             */
            if( boxtag == 0 || expr->intevaltag != boxtag )
            {
               expr->intevaltag = boxtag;
               SCIPintervalSetEntire(SCIP_INTERVAL_INFINITY, &expr->interval);
               expr->hastightened = FALSE;
               intersect = FALSE;
            }
            else
            {
               assert(expr->intevaltag == boxtag);

               /* the interval of the expression is valid, but did not change since the last call -> done with this expression */
               if( !expr->hastightened )
                  break;

               /* intersect result with existing interval */
               intersect = TRUE;
            }

            if( intersect )
            {
               /* start with interval that is stored in expression */
               interval = expr->interval;

               /* intersect with the interval of the auxiliary variable, if available */
               if( expr->auxvar != NULL )
               {
                  SCIP_Real lb = SCIPvarGetLbLocal(expr->auxvar);
                  SCIP_Real ub = SCIPvarGetUbLocal(expr->auxvar);
                  /* TODO use intevalvar? */
                  SCIP_Real inf = SCIPisInfinity(scip, -lb) ? -SCIP_INTERVAL_INFINITY : lb - SCIPepsilon(scip);
                  SCIP_Real sup = SCIPisInfinity(scip, ub) ? SCIP_INTERVAL_INFINITY : ub + SCIPepsilon(scip);
                  SCIP_INTERVAL auxinterval;

                  SCIPintervalSetBounds(&auxinterval, inf, sup);
                  SCIPintervalIntersect(&interval, interval, auxinterval);

                  /* check whether resulting interval is already empty */
                  if( SCIPintervalIsEmpty(SCIP_INTERVAL_INFINITY, interval) )
                  {
                     aborted = TRUE;
                     break;
                  }
               }
            }
            else
            {
               /* start with infinite interval [-inf,+inf] */
               SCIPintervalSetEntire(SCIP_INTERVAL_INFINITY, &interval);
            }

            if( expr->nenfos > 0 )
            {
               SCIP_CONSEXPR_NLHDLR* nlhdlr;
               SCIP_INTERVAL nlhdlrinterval;
               int e;

               /* for nodes with enforcement (having auxvar, thus during solve), nlhdlrs take care of interval evaluation */
               for( e = 0; e < expr->nenfos && !SCIPintervalIsEmpty(SCIP_INTERVAL_INFINITY, interval); ++e )
               {
                  nlhdlr = expr->enfos[e]->nlhdlr;
                  assert(nlhdlr != NULL);

                  /* skip nlhdlr if it does not provide interval evaluation */
                  if( !SCIPhasConsExprNlhdlrInteval(nlhdlr) )
                     continue;

                  /* let nlhdlr evaluate current expression */
                  nlhdlrinterval = interval;
                  SCIP_CALL( SCIPintevalConsExprNlhdlr(scip, nlhdlr, expr, expr->enfos[e]->nlhdlrexprdata, &nlhdlrinterval, intevalvar, intevalvardata) );
                  SCIPdebugMsg(scip, "computed interval [%g, %g] for expr ", nlhdlrinterval.inf, nlhdlrinterval.sup);
#ifdef SCIP_DEBUG
                  SCIP_CALL( SCIPprintConsExprExpr(scip, consexprhdlr, expr, NULL) );
                  SCIPdebugMsgPrint(scip, " (was [%g,%g]) by nlhdlr <%s>\n", expr->interval.inf, expr->interval.sup, nlhdlr->name);
#endif

                  /* intersect with interval */
                  SCIPintervalIntersect(&interval, interval, nlhdlrinterval);
               }
            }
            else
            {
               /* for node without enforcement (no auxvar, maybe in presolve), call the callback of the exprhdlr directly */
               SCIP_CALL( SCIPintevalConsExprExprHdlr(scip, expr, &interval, intevalvar, intevalvardata) );

#ifdef SCIP_DEBUG
               SCIPdebugMsg(scip, "computed interval [%g, %g] for expr ", interval.inf, interval.sup);
               SCIP_CALL( SCIPprintConsExprExpr(scip, consexprhdlr, expr, NULL) );
               SCIPdebugMsgPrint(scip, " (was [%g,%g]) by exprhdlr <%s>\n", expr->interval.inf, expr->interval.sup, expr->exprhdlr->name);
#endif
            }

            if( intersect )
            {
               /* make sure resulting interval is subset of expr->interval, if intersect is true
                * even though we passed expr->interval as input to the inteval callbacks,
                * these callbacks might not have taken it into account (most do not, actually)
                */
               SCIPintervalIntersect(&interval, interval, expr->interval);
            }

            /* check whether the resulting interval is empty */
            if( SCIPintervalIsEmpty(SCIP_INTERVAL_INFINITY, interval) )
            {
               SCIPintervalSetEmpty(&expr->interval);
               aborted = TRUE;
               break;
            }

            if( tightenauxvars )
            {
               /* tighten bounds of expression interval and the auxiliary variable */
               SCIP_CALL( SCIPtightenConsExprExprInterval(scip, expr, interval, force, NULL, &aborted, ntightenings) );

               if( aborted )
               {
                  SCIPintervalSetEmpty(&expr->interval);
                  break;
               }
            }
            else
            {
               /* update expression interval */
               SCIPintervalSetBounds(&expr->interval, interval.inf, interval.sup);
            }

            break;
         }

         default:
            /* you should never be here */
            SCIPABORT();
            break;
      }

      expr = SCIPexpriteratorGetNext(it);
   }

   /* evaluation leads to an empty interval -> detected infeasibility */
   if( aborted )
   {
      SCIPintervalSetEmpty(&rootexpr->interval);
      rootexpr->intevaltag = boxtag;

      if( infeasible != NULL)
         *infeasible = TRUE;
   }

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** propagates bounds for each sub-expression in the constraint by using variable bounds; the resulting bounds for the
 *  root expression will be intersected with the [lhs,rhs] which might lead to an empty interval
 */
static
SCIP_RETCODE forwardPropCons(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          conshdlr,         /**< constraint handler */
   SCIP_CONS*              cons,             /**< constraint to propagate */
   SCIP_Bool               force,            /**< force tightening even if below bound strengthening tolerance */
   unsigned int            boxtag,           /**< tag that uniquely identifies the current variable domains (with its values), or 0 */
   SCIP_Bool*              infeasible,       /**< buffer to store whether an expression's bounds were propagated to an empty interval */
   SCIP_Bool*              redundant,        /**< buffer to store whether the constraint is redundant */
   int*                    ntightenings      /**< buffer to store the number of auxiliary variable tightenings */
   )
{
   SCIP_INTERVAL interval;
   SCIP_CONSDATA* consdata;
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(infeasible != NULL);
   assert(redundant != NULL);
   assert(ntightenings != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   *infeasible = FALSE;
   *redundant = FALSE;
   *ntightenings = 0;

   /* propagate active and non-deleted constraints only */
   if( SCIPconsIsDeleted(cons) || !SCIPconsIsActive(cons) )
      return SCIP_OKAY;

   /* handle constant expressions separately; either the problem is infeasible or the constraint is redundant */
   if( consdata->expr->exprhdlr == SCIPgetConsExprExprHdlrValue(conshdlr) )
   {
      SCIP_Real value = SCIPgetConsExprExprValueValue(consdata->expr);
      if( (!SCIPisInfinity(scip, -consdata->lhs) && SCIPisFeasLT(scip, value - consdata->lhs, 0.0))
         || (!SCIPisInfinity(scip, consdata->rhs) && SCIPisFeasGT(scip, value - consdata->rhs, 0.0)) )
         *infeasible = TRUE;
      else
         *redundant = TRUE;

      return SCIP_OKAY;
   }

   /* use 0 tag to recompute intervals
    * we cannot trust variable bounds from SCIP, so relax them a little bit (a.k.a. epsilon)
    */
   SCIP_CALL( forwardPropExpr(scip, conshdlr, consdata->expr, force, TRUE, intEvalVarBoundTightening, (void*)SCIPconshdlrGetData(conshdlr), boxtag, infeasible, ntightenings) );

   /* it may happen that we detect infeasibility during forward propagation if we use previously computed intervals */
   if( !(*infeasible) )
   {
      /* relax sides by SCIPepsilon() and handle infinite sides */
      SCIP_Real lhs = SCIPisInfinity(scip, -consdata->lhs) ? -SCIP_INTERVAL_INFINITY : consdata->lhs - conshdlrdata->conssiderelaxamount;
      SCIP_Real rhs = SCIPisInfinity(scip,  consdata->rhs) ?  SCIP_INTERVAL_INFINITY : consdata->rhs + conshdlrdata->conssiderelaxamount;

      /* compare root expression interval with constraint sides; store the result in the root expression */
      SCIPintervalSetBounds(&interval, lhs, rhs);

      /* consider auxiliary variable stored in the root expression
       * it might happen that some other plug-ins tighten the bounds of these variables
       * we don't trust these bounds, so relax by epsilon
       */
      if( consdata->expr->auxvar != NULL )
      {
         SCIP_INTERVAL auxvarinterval;
         assert(SCIPvarGetLbLocal(consdata->expr->auxvar) <= SCIPvarGetUbLocal(consdata->expr->auxvar));  /* can SCIP ensure this by today? */

         SCIPintervalSetBounds(&auxvarinterval, SCIPvarGetLbLocal(consdata->expr->auxvar) - SCIPepsilon(scip),
            SCIPvarGetUbLocal(consdata->expr->auxvar) + SCIPepsilon(scip));
         SCIPintervalIntersect(&interval, interval, auxvarinterval);
      }

      SCIP_CALL( SCIPtightenConsExprExprInterval(scip, consdata->expr, interval, force, NULL, infeasible, ntightenings) );
   }

#ifdef SCIP_DEBUG
   if( *infeasible )
   {
      SCIPdebugMsg(scip, " -> found empty bound for an expression during forward propagation of constraint %s\n",
         SCIPconsGetName(cons));
   }
#endif

   return SCIP_OKAY;
}

/* export this function here, so it can be used by unittests but is not really part of the API */
/** propagates bounds for each sub-expression of a given set of constraints by starting from the root expressions; the
 *  expression will be traversed in breadth first search by using a queue
 *
 *  @note calling this function requires feasible intervals for each sub-expression; this is guaranteed by calling
 *  forwardPropCons() before calling this function
 */
static
SCIP_RETCODE reversePropConss(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONS**             conss,            /**< constraints to propagate */
   int                     nconss,           /**< total number of constraints to propagate */
   SCIP_Bool               force,            /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool               allexprs,         /**< whether reverseprop should be called for all expressions, regardless of whether their interval was tightened */
   SCIP_Bool*              infeasible,       /**< buffer to store whether an expression's bounds were propagated to an empty interval */
   int*                    ntightenings      /**< buffer to store the number of (variable) tightenings */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_QUEUE* queue;
   int i;

   assert(scip != NULL);
   assert(conss != NULL);
   assert(nconss >= 0);
   assert(infeasible != NULL);
   assert(ntightenings != NULL);

   *infeasible = FALSE;
   *ntightenings = 0;

   if( nconss == 0 )
      return SCIP_OKAY;

   /* create queue */
   SCIP_CALL( SCIPqueueCreate(&queue, SCIPgetNVars(scip), 2.0) );

   /* add root expressions to the queue */
   for( i = 0; i < nconss; ++i )
   {
      assert(conss[i] != NULL);
      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      /* propagate active and non-deleted constraints only */
      if( SCIPconsIsDeleted(conss[i]) || !SCIPconsIsActive(conss[i]) )
         continue;

      /* skip expressions that could not have been tightened, unless allexprs is set */
      if( !consdata->expr->hastightened && !allexprs )
         continue;

      /* add expressions which are not in the queue so far */
      if( !consdata->expr->inqueue && SCIPgetConsExprExprNChildren(consdata->expr) > 0 )
      {
         SCIP_CALL( SCIPqueueInsert(queue, (void*) consdata->expr) );
         consdata->expr->inqueue = TRUE;
      }
   }

   /* main loop that calls reverse propagation for expressions on the queue
    * when reverseprop finds a tightening for an expression, then that expression is added to the queue (within the reverseprop call)
    */
   while( !SCIPqueueIsEmpty(queue) && !(*infeasible) )
   {
      SCIP_CONSEXPR_EXPR* expr;
      int e;

      expr = (SCIP_CONSEXPR_EXPR*) SCIPqueueRemove(queue);
      assert(expr != NULL);

      /* mark that the expression is not in the queue anymore */
      expr->inqueue = FALSE;

      if( expr->nenfos > 0 )
      {
         /* for nodes with enforcement, call reverse propagation callbacks of nlhdlrs */
         for( e = 0; e < expr->nenfos && !*infeasible; ++e )
         {
            SCIP_CONSEXPR_NLHDLR* nlhdlr;
            int nreds;

            nlhdlr = expr->enfos[e]->nlhdlr;
            assert(nlhdlr != NULL);

            /* call the reverseprop of the nlhdlr */
#ifdef SCIP_DEBUG
            SCIPdebugMsg(scip, "call reverse propagation for ");
            SCIP_CALL( SCIPprintConsExprExpr(scip, SCIPfindConshdlr(scip, CONSHDLR_NAME), expr, NULL) );
            SCIPdebugMsgPrint(scip, " in [%g,%g] using nlhdlr <%s>\n", expr->interval.inf, expr->interval.sup, nlhdlr->name);
#endif

            nreds = 0;
            SCIP_CALL( SCIPreversepropConsExprNlhdlr(scip, nlhdlr, expr, expr->enfos[e]->nlhdlrexprdata, queue, infeasible, &nreds, force) );
            assert(nreds >= 0);
            *ntightenings += nreds;
         }
      }
      else
      {
         /* if node without enforcement (no auxvar or in presolve), call reverse propagation callback of exprhdlr directly */
         int nreds = 0;

#ifdef SCIP_DEBUG
         SCIPdebugMsg(scip, "call reverse propagation for ");
         SCIP_CALL( SCIPprintConsExprExpr(scip, SCIPfindConshdlr(scip, CONSHDLR_NAME), expr, NULL) );
         SCIPdebugMsgPrint(scip, " in [%g,%g] using exprhdlr <%s>\n", expr->interval.inf, expr->interval.sup, expr->exprhdlr->name);
#endif

         /* call the reverseprop of the exprhdlr */
         SCIP_CALL( SCIPreversepropConsExprExprHdlr(scip, expr, queue, infeasible, &nreds, force) );
         assert(nreds >= 0);
         *ntightenings += nreds;
      }

      /* if allexprs is set, then make sure that all children of expr with children are in the queue
       * SCIPtightenConsExprExpr only adds children to the queue which have reverseprop capability
       */
      if( allexprs )
         for( i = 0; i < SCIPgetConsExprExprNChildren(expr); ++i )
         {
            SCIP_CONSEXPR_EXPR* child;

            child = SCIPgetConsExprExprChildren(expr)[i];

            if( !child->inqueue && SCIPgetConsExprExprNChildren(child) > 0 )
            {
               SCIP_CALL( SCIPqueueInsert(queue, (void*) child) );
               child->inqueue = TRUE;
            }
         }

      /* stop propagation if the problem is infeasible */
      if( *infeasible )
         break;
   }

   /* reset expr->inqueue for all remaining expr's in queue (can happen in case of early stop due to infeasibility) */
   while( !SCIPqueueIsEmpty(queue) )
   {
      SCIP_CONSEXPR_EXPR* expr;

      expr = (SCIP_CONSEXPR_EXPR*) SCIPqueueRemove(queue);

      /* mark that the expression is not in the queue anymore */
      expr->inqueue = FALSE;
   }

   /* free the queue */
   SCIPqueueFree(&queue);

   return SCIP_OKAY;
}

/** calls domain propagation for a given set of constraints; the algorithm alternates calls of forward and reverse
 *  propagation; the latter only for nodes which have been tightened during the propagation loop;
 *
 *  the propagation algorithm works as follows:
 *
 *   0.) mark all expressions as non-tightened
 *
 *   1.) apply forward propagation and intersect the root expressions with the constraint sides; mark root nodes which
 *       have been changed after intersecting with the constraint sides
 *
 *   2.) apply reverse propagation to each root expression which has been marked as tightened; don't explore
 *       sub-expressions which have not changed since the beginning of the propagation loop
 *
 *   3.) if we have found enough tightenings go to 1.) otherwise leave propagation loop
 *
 *  @note after calling forward propagation for a constraint we mark this constraint as propagated; this flag might be
 *  reset during the reverse propagation when we find a bound tightening of a variable expression contained in the
 *  constraint; resetting this flag is done in the EVENTEXEC callback of the event handler
 *
 *  @note when using forward and reverse propagation alternatingly we reuse expression intervals computed in previous
 *  iterations
 */
static
SCIP_RETCODE propConss(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONS**           conss,              /**< constraints to propagate */
   int                   nconss,             /**< total number of constraints */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_RESULT*          result,             /**< pointer to store the result */
   int*                  nchgbds,            /**< buffer to add the number of changed bounds */
   int*                  ndelconss           /**< buffer to add the number of deleted constraints */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP_Bool cutoff;
   SCIP_Bool redundant;
   SCIP_Bool success;
   int ntightenings;
   int roundnr;
   int i;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(conss != NULL);
   assert(nconss >= 0);
   assert(result != NULL);
   assert(nchgbds != NULL);
   assert(*nchgbds >= 0);
   assert(ndelconss != NULL);

   /* no constraints to propagate */
   if( nconss == 0 )
   {
      *result = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
   }

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   *result = SCIP_DIDNOTFIND;
   roundnr = 0;
   cutoff = FALSE;

   /* increase lastinteval tag */
   ++(conshdlrdata->lastintevaltag);
   assert(conshdlrdata->lastintevaltag > 0);

   /* main propagation loop */
   do
   {
      SCIPdebugMsg(scip, "start propagation round %d\n", roundnr);

      /* apply forward propagation; recompute expression intervals if it is called for the first time (this also marks
       * all expressions as non-tightened)
       */
      for( i = 0; i < nconss; ++i )
      {
         consdata = SCIPconsGetData(conss[i]);
         assert(consdata != NULL);

         /* in the first round, we reevaluate all bounds to remove some possible leftovers that could be in this
          * expression from a reverse propagation in a previous propagation round
          */
         if( !SCIPconsIsDeleted(conss[i]) && SCIPconsIsActive(conss[i]) && (!consdata->ispropagated || roundnr == 0) )
         {
            SCIPdebugMsg(scip, "call forwardPropCons() for constraint <%s> (round %d): ", SCIPconsGetName(conss[i]), roundnr);
            SCIPdebugPrintCons(scip, conss[i], NULL);

            cutoff = FALSE;
            redundant = FALSE;
            ntightenings = 0;

            SCIP_CALL( forwardPropCons(scip, conshdlr, conss[i], force, conshdlrdata->lastintevaltag, &cutoff,
               &redundant, &ntightenings) );
            assert(ntightenings >= 0);
            *nchgbds += ntightenings;

            if( cutoff )
            {
               SCIPdebugMsg(scip, " -> cutoff\n");
               *result = SCIP_CUTOFF;
               return SCIP_OKAY;
            }
            if( ntightenings > 0 )
               *result = SCIP_REDUCEDDOM;
            if( redundant )
               *ndelconss += 1;

            /* mark constraint as propagated; this will be reset via the event system when we find a variable tightening */
            consdata->ispropagated = TRUE;
         }
      }

      /* apply backward propagation; mark constraint as propagated */
      /* TODO during presolve, maybe we want to run reverseprop for ALL expressions once, if roundnr==0 ? */
      SCIP_CALL( reversePropConss(scip, conss, nconss, force, FALSE, &cutoff, &ntightenings) );

      /* @todo add parameter for the minimum number of tightenings to trigger a new propagation round */
      success = ntightenings > 0;

      if( nchgbds != NULL )
         *nchgbds += ntightenings;

      if( cutoff )
      {
         SCIPdebugMsg(scip, " -> cutoff\n");
         *result = SCIP_CUTOFF;
         return SCIP_OKAY;
      }

      if( success )
         *result = SCIP_REDUCEDDOM;
   }
   while( success && ++roundnr < conshdlrdata->maxproprounds );

   return SCIP_OKAY;
}

/** checks constraints for redundancy
 *
 * Checks whether the activity of constraint functions is a subset of the constraint sides (relaxed by feastol).
 * To compute the activity, we use forwardPropCons(), but relax variable bounds by feastol, because solutions to be checked
 * might violate variable bounds by up to feastol, too.
 * This is the main reason why the redundancy check is not done in propConss(), which relaxes variable bounds by epsilon only.
 *
 * Also removes constraints of the form lhs <= variable <= rhs.
 *
 * @TODO it would be sufficient to check constraints for which we know that they are not currently violated by a valid solution
 */
static
SCIP_RETCODE checkRedundancyConss(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONS**           conss,              /**< constraints to propagate */
   int                   nconss,             /**< total number of constraints */
   SCIP_Bool*            cutoff,             /**< pointer to store whether infeasibility has been identified */
   int*                  ndelconss,          /**< buffer to add the number of deleted constraints */
   int*                  nchgbds             /**< buffer to add the number of variable bound tightenings */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP_INTERVAL activity;
   SCIP_INTERVAL sides;
   int i;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(conss != NULL);
   assert(nconss >= 0);
   assert(cutoff != NULL);
   assert(ndelconss != NULL);
   assert(nchgbds != NULL);

   /* no constraints to check */
   if( nconss == 0 )
      return SCIP_OKAY;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* increase lastinteval tag */
   ++(conshdlrdata->lastintevaltag);
   assert(conshdlrdata->lastintevaltag > 0);

   SCIPdebugMsg(scip, "checking %d constraints for redundancy\n", nconss);

   *cutoff = FALSE;
   for( i = 0; i < nconss; ++i )
   {
      if( !SCIPconsIsActive(conss[i]) || SCIPconsIsDeleted(conss[i]) )
         continue;

      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      /* handle constant expressions separately: either the problem is infeasible or the constraint is redundant */
      if( consdata->expr->exprhdlr == SCIPgetConsExprExprHdlrValue(conshdlr) )
      {
         SCIP_Real value = SCIPgetConsExprExprValueValue(consdata->expr);

         if(  (!SCIPisInfinity(scip, -consdata->lhs) && value < consdata->lhs - SCIPfeastol(scip))
            || (!SCIPisInfinity(scip, consdata->rhs) && value > consdata->rhs + SCIPfeastol(scip)) )
         {
            SCIPdebugMsg(scip, "constant constraint <%s> is infeasible: %g in [%g,%g] ", SCIPconsGetName(conss[i]), value, consdata->lhs, consdata->rhs);
            *cutoff = TRUE;

            return SCIP_OKAY;
         }

         SCIPdebugMsg(scip, "constant constraint <%s> is redundant: %g in [%g,%g] ", SCIPconsGetName(conss[i]), value, consdata->lhs, consdata->rhs);

         SCIP_CALL( SCIPdelConsLocal(scip, conss[i]) );
         ++*ndelconss;

         continue;
      }

      /* handle variable expressions separately: tighten variable bounds to constraint sides, then remove constraint (now redundant) */
      if( consdata->expr->exprhdlr == SCIPgetConsExprExprHdlrVar(conshdlr) )
      {
         SCIP_VAR* var;
         SCIP_Bool tightened;

         var = SCIPgetConsExprExprVarVar(consdata->expr);
         assert(var != NULL);

         SCIPdebugMsg(scip, "variable constraint <%s> can be made redundant: <%s>[%g,%g] in [%g,%g] ", SCIPconsGetName(conss[i]), SCIPvarGetName(var), SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var), consdata->lhs, consdata->rhs);

         /* ensure that variable bounds are within constraint sides */
         if( !SCIPisInfinity(scip, -consdata->lhs) )
         {
            SCIP_CALL( SCIPtightenVarLb(scip, var, consdata->lhs, TRUE, cutoff, &tightened) );

            if( tightened )
               ++*nchgbds;

            if( *cutoff )
               return SCIP_OKAY;
         }

         if( !SCIPisInfinity(scip, consdata->rhs) )
         {
            SCIP_CALL( SCIPtightenVarUb(scip, var, consdata->rhs, TRUE, cutoff, &tightened) );

            if( tightened )
               ++*nchgbds;

            if( *cutoff )
               return SCIP_OKAY;
         }

         /* delete the (now) redundant constraint locally */
         SCIP_CALL( SCIPdelConsLocal(scip, conss[i]) );
         ++*ndelconss;

         continue;
      }

      /* reevaluate all bounds to remove some possible leftovers that could be in this
       * expression from a reverse propagation in a previous propagation round
       *
       * we relax variable bounds by feastol here, as solutions that are checked later can also violate
       * variable bounds by up to feastol
       * (relaxing fixed variables seems to be too much, but they would be removed by presolve soon anyway)
       */
      SCIPdebugMsg(scip, "call forwardPropExpr() for constraint <%s>: ", SCIPconsGetName(conss[i]));
      SCIPdebugPrintCons(scip, conss[i], NULL);

      SCIP_CALL( forwardPropExpr(scip, conshdlr, consdata->expr, FALSE, FALSE, intEvalVarRedundancyCheck, NULL, conshdlrdata->lastintevaltag, cutoff, NULL) );

      /* it is unlikely that we detect infeasibility by doing forward propagation */
      if( *cutoff )
      {
         SCIPdebugMsg(scip, " -> cutoff\n");
         return SCIP_OKAY;
      }

      assert(consdata->expr->intevaltag == conshdlrdata->lastintevaltag);
      activity = consdata->expr->interval;

      /* relax sides by feastol
       * we could accept every solution that violates constraints up to feastol as redundant, so this is the most permissive we can be
       */
      SCIPintervalSetBounds(&sides,
         SCIPisInfinity(scip, -consdata->lhs) ? -SCIP_INTERVAL_INFINITY : consdata->lhs - SCIPfeastol(scip),
         SCIPisInfinity(scip,  consdata->rhs) ?  SCIP_INTERVAL_INFINITY : consdata->rhs + SCIPfeastol(scip));

      if( SCIPintervalIsSubsetEQ(SCIP_INTERVAL_INFINITY, activity, sides) )
      {
         SCIPdebugMsg(scip, " -> redundant: activity [%g,%g] within sides [%g,%g]\n", activity.inf, activity.sup, consdata->lhs, consdata->rhs);

         SCIP_CALL( SCIPdelConsLocal(scip, conss[i]) );
         ++*ndelconss;

         return SCIP_OKAY;
      }

      SCIPdebugMsg(scip, " -> not redundant: activity [%g,%g] not within sides [%g,%g]\n", activity.inf, activity.sup, consdata->lhs, consdata->rhs);
   }

   return SCIP_OKAY;
}

/** install nlhdlrs in one expression */
static
SCIP_RETCODE detectNlhdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression for which to run detection routines */
   SCIP_Bool             isroot,             /**< expression is a root expression, that is, it defines a constraint */
   SCIP_CONSEXPR_NLHDLR**  nlhdlrssuccess,   /**< buffer for nlhdlrs that had success detecting structure at expression */
   SCIP_CONSEXPR_NLHDLREXPRDATA** nlhdlrssuccessexprdata, /**< buffer for exprdata of nlhdlrs */
   SCIP_Bool*            infeasible          /**< buffer to indicate whether infeasibility has been detected */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_Bool enforcedbelow;
   SCIP_Bool enforcedabove;
   SCIP_CONSEXPR_EXPRENFO_METHOD enforcemethods;
   SCIP_Bool nlhdlrenforcedbelow;
   SCIP_Bool nlhdlrenforcedabove;
   SCIP_CONSEXPR_EXPRENFO_METHOD nlhdlrenforcemethods;
   SCIP_Bool success;
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata;
   int ntightenings;
   int nsuccess;
   int e, h;

   assert(conshdlr != NULL);
   assert(expr != NULL);
   assert(nlhdlrssuccess != NULL);
   assert(nlhdlrssuccessexprdata != NULL);
   assert(infeasible != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->auxvarid >= 0);

   /* there should be no enforcer yet, i.e., detection should not have been run already */
   assert(expr->nenfos == 0);
   assert(expr->enfos == NULL);

   /* analyze expression with nonlinear handlers
    * if nobody positively (up) locks expr -> only need to enforce expr >= auxvar -> no need for underestimation
    * if nobody negatively (down) locks expr -> only need to enforce expr <= auxvar -> no need for overestimation
    */
   nsuccess = 0;
   enforcemethods = SCIP_CONSEXPR_EXPRENFO_NONE;
   enforcedbelow = (SCIPgetConsExprExprNLocksPos(expr) == 0); /* no need for underestimation */
   enforcedabove = (SCIPgetConsExprExprNLocksNeg(expr) == 0); /* no need for overestimation */

   SCIPdebugMsg(scip, "detecting nlhdlrs for %s expression %p (%s); start with below %d above %d\n",
      isroot ? "root" : "non-root", (void*)expr, SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), enforcedbelow, enforcedabove);

   for( h = 0; h < conshdlrdata->nnlhdlrs && !*infeasible; ++h )
   {
      SCIP_CONSEXPR_NLHDLR* nlhdlr;
      SCIP_INTERVAL interval;

      nlhdlr = conshdlrdata->nlhdlrs[h];
      assert(nlhdlr != NULL);

      /* skip disabled nlhdlrs */
      if( !nlhdlr->enabled )
         continue;

      /* call detect routine of nlhdlr */
      nlhdlrexprdata = NULL;
      success = FALSE;
      nlhdlrenforcemethods = enforcemethods;
      nlhdlrenforcedbelow = enforcedbelow;
      nlhdlrenforcedabove = enforcedabove;
      SCIP_CALL( SCIPdetectConsExprNlhdlr(scip, conshdlr, nlhdlr, expr, isroot, &nlhdlrenforcemethods, &nlhdlrenforcedbelow, &nlhdlrenforcedabove, &success, &nlhdlrexprdata) );

      /* detection is only allowed to augment to the various parameters (enforce "more", add "more" methods) */
      assert(nlhdlrenforcemethods >= enforcemethods);
      assert(nlhdlrenforcedbelow >= enforcedbelow);
      assert(nlhdlrenforcedabove >= enforcedabove);

      if( !success )
      {
         /* nlhdlrexprdata can only be non-NULL if it provided some functionality */
         assert(nlhdlrexprdata == NULL);
         assert(nlhdlrenforcemethods == enforcemethods);
         assert(nlhdlrenforcedbelow == enforcedbelow);
         assert(nlhdlrenforcedabove == enforcedabove);

         continue;
      }

      SCIPdebugMsg(scip, "nlhdlr <%s> detect successful; now enforced below: %d above: %d methods: %d\n",
         SCIPgetConsExprNlhdlrName(nlhdlr), nlhdlrenforcedbelow, nlhdlrenforcedabove, nlhdlrenforcemethods);

      /* if the nlhdlr enforces, then it must have added at least one enforcement method */
      assert(nlhdlrenforcemethods > enforcemethods || (nlhdlrenforcedbelow == enforcedbelow && nlhdlrenforcedabove == enforcedabove));

      /* remember nlhdlr and its data */
      nlhdlrssuccess[nsuccess] = nlhdlr;
      nlhdlrssuccessexprdata[nsuccess] = nlhdlrexprdata;
      ++nsuccess;

      /* update enforcement flags */
      enforcemethods = nlhdlrenforcemethods;
      enforcedbelow = nlhdlrenforcedbelow;
      enforcedabove = nlhdlrenforcedabove;

      if( SCIPgetStage(scip) == SCIP_STAGE_SOLVING )
      {
         /* let nlhdlr evaluate current expression
          * we do this here, because we want to call reverseprop after detect (if called from initlp),
          * but some nlhdlr (i.e., quadratic) require that its inteval has been called before
          */
         interval = expr->interval;
         SCIP_CALL( SCIPintevalConsExprNlhdlr(scip, nlhdlr, expr, nlhdlrexprdata, &interval, intEvalVarBoundTightening, (void*)SCIPconshdlrGetData(conshdlr)) );
         SCIPdebugMsg(scip, "nlhdlr <%s> computed interval [%g,%g]\n", SCIPgetConsExprNlhdlrName(nlhdlr), interval.inf, interval.sup);
         /* tighten bounds of expression interval and the auxiliary variable */
         SCIP_CALL( SCIPtightenConsExprExprInterval(scip, expr, expr->interval, TRUE, NULL, infeasible, &ntightenings) );
      }
   }

   /* stop if the expression cannot be enforced but we are already in solving stage
    * (as long as the expression provides its callbacks, the default nlhdlr should have provided all enforcement methods)
    */
   if( (!enforcedbelow || !enforcedabove) && !*infeasible && SCIPgetStage(scip) == SCIP_STAGE_SOLVING )
   {
      SCIPerrorMessage("no nonlinear handler provided enforcement for %s expression %s auxvar\n",
         SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)),
         (!enforcedbelow && !enforcedabove) ? "==" : (!enforcedbelow ? "<=" : ">="));
      return SCIP_ERROR;
   }

   if( nsuccess == 0 )
      return SCIP_OKAY;

   /* copy collected nlhdlrs into expr->enfos */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &expr->enfos, nsuccess) );
   for( e = 0; e < nsuccess; ++e )
   {
      SCIP_CALL( SCIPallocBlockMemory(scip, &expr->enfos[e]) );  /*lint !e866 */
      expr->enfos[e]->nlhdlr = nlhdlrssuccess[e];
      expr->enfos[e]->nlhdlrexprdata = nlhdlrssuccessexprdata[e];
      expr->enfos[e]->issepainit = FALSE;
   }
   expr->nenfos = nsuccess;

   return SCIP_OKAY;
}

/** detect nlhdlrs that can handle the expressions */
static
SCIP_RETCODE detectNlhdlrs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONS**           conss,              /**< constraints to check for auxiliary variables */
   int                   nconss,             /**< total number of constraints */
   SCIP_Bool*            infeasible          /**< pointer to store whether an infeasibility was detected while creating the auxiliary vars */
   )
{
   SCIP_CONSEXPR_NLHDLR** nlhdlrssuccess;   /* buffer for nlhdlrs that had success detecting structure at expression */
   SCIP_CONSEXPR_NLHDLREXPRDATA** nlhdlrssuccessexprdata; /* buffer for exprdata of nlhdlrs */
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP_CONSEXPR_EXPR* expr;
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_Bool redundant;
   int ntightenings;
   int i;

   assert(conss != NULL || nconss == 0);
   assert(nconss >= 0);
   assert(infeasible != NULL);
   assert(SCIPgetStage(scip) == SCIP_STAGE_PRESOLVING || SCIPgetStage(scip) == SCIP_STAGE_SOLVING);  /* should only be called in presolve or initlp */

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );

   /* allocate some buffer for temporary storage of nlhdlr detect result */
   SCIP_CALL( SCIPallocBufferArray(scip, &nlhdlrssuccess, conshdlrdata->nnlhdlrs) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nlhdlrssuccessexprdata, conshdlrdata->nnlhdlrs) );

   if( SCIPgetStage(scip) == SCIP_STAGE_SOLVING )
   {
      /* increase lastinteval tag */
      ++(conshdlrdata->lastintevaltag);
      assert(conshdlrdata->lastintevaltag > 0);
   }

   *infeasible = FALSE;
   for( i = 0; i < nconss; ++i )
   {
      assert(conss != NULL && conss[i] != NULL);

      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);
      assert(consdata->expr != NULL);

      if( SCIPgetStage(scip) == SCIP_STAGE_SOLVING )
      {
         /* make sure intervals in expression are uptodate (use 0 force recomputing)
          * we do this here to have bounds for the auxiliary variables and for a reverseprop call at the end
          * we don't do auxiliary variables if in presolve, so do only in solving
          */
         SCIP_CALL( forwardPropCons(scip, conshdlr, conss[i], FALSE, conshdlrdata->lastintevaltag, infeasible, &redundant, &ntightenings) );
         if( *infeasible )
         {
            SCIPdebugMsg(scip, "infeasibility detected in forward prop of constraint <%s>\n", SCIPconsGetName(conss[i]));
            break;
         }

         /* forwardPropCons recognized redundant if the cons consists of a value expression
          * for that one, we don't need nlhdlrs
          * TODO can we delete constraint here (we are in initlp) ?
          */
         if( redundant )
            continue;

#ifdef WITH_DEBUG_SOLUTION
         if( SCIPdebugIsMainscip(scip) )
         {
            SCIP_SOL* debugsol;

            SCIP_CALL( SCIPdebugGetSol(scip, &debugsol) );

            if( debugsol != NULL ) /* it can be compiled WITH_DEBUG_SOLUTION, but still no solution given */
            {
               /* evaluate expression in debug solution, so we can set the solution value of created auxiliary variables
                * in SCIPcreateConsExprExprAuxVar()
                */
               SCIP_CALL( SCIPevalConsExprExpr(scip, conshdlr, consdata->expr, debugsol, 0) );
            }
         }
#endif
      }

      /* compute integrality information for all subexpressions */
      SCIP_CALL( SCIPcomputeConsExprExprIntegral(scip, conshdlr, consdata->expr) );

      if( SCIPgetStage(scip) == SCIP_STAGE_SOLVING )
      {
         /* ensure auxiliary variable for root expression exists (not always necessary, but it simplifies things) */
         SCIP_CALL( SCIPcreateConsExprExprAuxVar(scip, conshdlr, consdata->expr, NULL) );
         assert(consdata->expr->auxvar != NULL);  /* couldn't this fail if the expression is only a variable? */

         /* change the bounds of the auxiliary variable of the root node to [lhs,rhs] */
         SCIP_CALL( SCIPtightenVarLb(scip, consdata->expr->auxvar, consdata->lhs, FALSE, infeasible, NULL) );
         if( *infeasible )
         {
            SCIPdebugMsg(scip, "infeasibility detected while creating vars: lhs of constraint (%g) > ub of node (%g)\n",
               consdata->lhs, SCIPvarGetUbLocal(consdata->expr->auxvar));
            break;
         }

         SCIP_CALL( SCIPtightenVarUb(scip, consdata->expr->auxvar, consdata->rhs, FALSE, infeasible, NULL) );
         if( *infeasible )
         {
            SCIPdebugMsg(scip, "infeasibility detected while creating vars: rhs of constraint (%g) < lb of node (%g)\n",
               consdata->rhs, SCIPvarGetLbLocal(consdata->expr->auxvar));
            break;
         }
      }

      SCIP_CALL( SCIPexpriteratorInit(it, consdata->expr, SCIP_CONSEXPRITERATOR_DFS, TRUE) );  /* TODO init once for all conss */
      expr = SCIPexpriteratorGetCurrent(it);
      while( !SCIPexpriteratorIsEnd(it) )
      {
         if( expr->nenfos > 0 )
         {
            /* because of common sub-expressions it might happen that we already detected a nonlinear handler and added it to the expr
             * then also the subtree has been investigated already and we can stop iterating further down
             * HOWEVER: most likely we have been running DETECT with isroot=FALSE, which may interest less nlhdlrs
             * thus, if expr is the root expression, then rerun DETECT
             */
            if( expr == consdata->expr )
            {
               SCIP_CALL( freeEnfoData(scip, expr, FALSE) );
            }
            else
            {
               expr = SCIPexpriteratorSkipDFS(it);
               continue;
            }
         }

         /* during solve: if there is an auxiliary variable here, then there is some-one requiring that
          *   an auxvar equals (or approximates) to value of this expression or we are at the root expression (expr==consdata->expr)
          *   thus, we need to find nlhdlrs
          * during presolve: we do detect for all expressions for now, expecting the handler to only become
          *   active if they want to contribute in propagation
          */
         if( expr->auxvar != NULL || SCIPgetStage(scip) == SCIP_STAGE_PRESOLVING )
         {
            SCIP_CALL( detectNlhdlr(scip, conshdlr, expr, expr == consdata->expr, nlhdlrssuccess, nlhdlrssuccessexprdata, infeasible) );

            if( *infeasible )
               break;
         }

         expr = SCIPexpriteratorGetNext(it);
      }

      if( *infeasible )
      {
         SCIPdebugMsg(scip, "infeasibility detected while detecting nlhdlr\n");
         break;
      }
   }

   SCIPexpriteratorFree(&it);
   SCIPfreeBufferArray(scip, &nlhdlrssuccessexprdata);
   SCIPfreeBufferArray(scip, &nlhdlrssuccess);

   if( SCIPgetStage(scip) == SCIP_STAGE_SOLVING )
   {
      /* call reverse propagation for ALL expressions
       * This can ensure that auxiliary variables take only values that are within the domain of functions that use them
       * for example, sqrt(x) in [-infty,infty] will ensure x >= 0, thus regardless of [-infty,infty] being pretty useless.
       * Also we do this already here because LP solving and separation will be called next, which could already profit
       * from the tighter bounds (or: cons_expr_pow spits out a warning in separation if the child can be negative and exponent not integral).
       */
      SCIP_CALL( reversePropConss(scip, conss, nconss, FALSE, TRUE, infeasible, &ntightenings) );
   }

   return SCIP_OKAY;
}

/** stores all variable expressions into a given constraint */
static
SCIP_RETCODE storeVarExprs(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          conshdlr,         /**< constraint handler */
   SCIP_CONSDATA*          consdata          /**< constraint data */
   )
{
   assert(consdata != NULL);

   /* skip if we have stored the variable expressions already*/
   if( consdata->varexprs != NULL )
      return SCIP_OKAY;

   assert(consdata->varexprs == NULL);
   assert(consdata->nvarexprs == 0);

   /* create array to store all variable expressions; the number of variable expressions is bounded by SCIPgetNTotalVars() */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &consdata->varexprs, SCIPgetNTotalVars(scip)) );

   SCIP_CALL( SCIPgetConsExprExprVarExprs(scip, conshdlr, consdata->expr, consdata->varexprs, &(consdata->nvarexprs)) );
   assert(SCIPgetNTotalVars(scip) >= consdata->nvarexprs);

   /* realloc array if there are less variable expression than variables */
   if( SCIPgetNTotalVars(scip) > consdata->nvarexprs )
   {
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &consdata->varexprs, SCIPgetNTotalVars(scip), consdata->nvarexprs) );
   }

   return SCIP_OKAY;
}

/** frees all variable expression stored in storeVarExprs() */
static
SCIP_RETCODE freeVarExprs(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSDATA*          consdata          /**< constraint data */
   )
{
   int i;

   assert(consdata != NULL);

   /* skip if we have stored the variable expressions already*/
   if( consdata->varexprs == NULL )
      return SCIP_OKAY;

   assert(consdata->varexprs != NULL);
   assert(consdata->nvarexprs >= 0);

   /* release variable expressions */
   for( i = 0; i < consdata->nvarexprs; ++i )
   {
      assert(consdata->varexprs[i] != NULL);
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &consdata->varexprs[i]) );
      assert(consdata->varexprs[i] == NULL);
   }

   /* free variable expressions */
   SCIPfreeBlockMemoryArrayNull(scip, &consdata->varexprs, consdata->nvarexprs);
   consdata->varexprs = NULL;
   consdata->nvarexprs = 0;

   return SCIP_OKAY;
}

/** computes violation of a constraint */
static
SCIP_RETCODE computeViolation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_SOL*             sol,                /**< solution or NULL if LP solution should be used */
   unsigned int          soltag              /**< tag that uniquely identifies the solution (with its values), or 0. */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_Real activity;

   assert(scip != NULL);
   assert(cons != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   SCIP_CALL( SCIPevalConsExprExpr(scip, SCIPconsGetHdlr(cons), consdata->expr, sol, soltag) );
   activity = SCIPgetConsExprExprValue(consdata->expr);

   /* consider constraint as violated if it is undefined in the current point */
   if( activity == SCIP_INVALID ) /*lint !e777*/
   {
      consdata->lhsviol = SCIPinfinity(scip);
      consdata->rhsviol = SCIPinfinity(scip);
      return SCIP_OKAY;
   }

   /* compute violations */
   consdata->lhsviol = SCIPisInfinity(scip, -consdata->lhs) ? -SCIPinfinity(scip) : consdata->lhs  - activity;
   consdata->rhsviol = SCIPisInfinity(scip,  consdata->rhs) ? -SCIPinfinity(scip) : activity - consdata->rhs;

   return SCIP_OKAY;
}

/** catch variable events */
static
SCIP_RETCODE catchVarEvents(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler */
   SCIP_CONS*            cons                /**< constraint for which to catch bound change events */
   )
{
   SCIP_EVENTTYPE eventtype;
   SCIP_CONSDATA* consdata;
   SCIP_VAR* var;
   int i;

   assert(eventhdlr != NULL);
   assert(cons != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   assert(consdata->varexprs != NULL);
   assert(consdata->nvarexprs >= 0);

   /* check if we have catched variable events already */
   if( consdata->vareventdata != NULL )
      return SCIP_OKAY;

   assert(consdata->vareventdata == NULL);

   SCIPdebugMsg(scip, "catchVarEvents for %s\n", SCIPconsGetName(cons));

   eventtype = SCIP_EVENTTYPE_BOUNDCHANGED | SCIP_EVENTTYPE_VARFIXED;

   /* allocate enough memory to store all event data structs */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &consdata->vareventdata, consdata->nvarexprs) );

   for( i = 0; i < consdata->nvarexprs; ++i )
   {
      assert(consdata->varexprs[i] != NULL);
      assert(SCIPisConsExprExprVar(consdata->varexprs[i]));

      var = SCIPgetConsExprExprVarVar(consdata->varexprs[i]);
      assert(var != NULL);

      SCIP_CALL( SCIPallocBlockMemory(scip, &(consdata->vareventdata[i])) ); /*lint !e866*/
      consdata->vareventdata[i]->cons = cons;
      consdata->vareventdata[i]->varexpr = consdata->varexprs[i];

      SCIP_CALL( SCIPcatchVarEvent(scip, var, eventtype, eventhdlr, (SCIP_EVENTDATA*) consdata->vareventdata[i],
            &(consdata->vareventdata[i]->filterpos)) );
   }

   return SCIP_OKAY;
}

/** drop variable events */
static
SCIP_RETCODE dropVarEvents(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler */
   SCIP_CONS*            cons                /**< constraint for which to drop bound change events */
   )
{
   SCIP_EVENTTYPE eventtype;
   SCIP_CONSDATA* consdata;
   SCIP_VAR* var;
   int i;

   assert(eventhdlr != NULL);
   assert(cons != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* check if we have catched variable events already */
   if( consdata->vareventdata == NULL )
      return SCIP_OKAY;

   assert(consdata->varexprs != NULL);
   assert(consdata->nvarexprs >= 0);
   assert(consdata->vareventdata != NULL);

   eventtype = SCIP_EVENTTYPE_BOUNDCHANGED | SCIP_EVENTTYPE_VARFIXED;

   SCIPdebugMsg(scip, "dropVarEvents for %s\n", SCIPconsGetName(cons));

   for( i = consdata->nvarexprs - 1; i >= 0; --i )
   {
      var = SCIPgetConsExprExprVarVar(consdata->varexprs[i]);
      assert(var != NULL);

      assert(SCIPgetConsExprExprVarVar(consdata->vareventdata[i]->varexpr) == var);
      assert(consdata->vareventdata[i]->cons == cons);
      assert(consdata->vareventdata[i]->varexpr == consdata->varexprs[i]);
      assert(consdata->vareventdata[i]->filterpos >= 0);

      SCIP_CALL( SCIPdropVarEvent(scip, var, eventtype, eventhdlr, (SCIP_EVENTDATA*) consdata->vareventdata[i], consdata->vareventdata[i]->filterpos) );

      SCIPfreeBlockMemory(scip, &consdata->vareventdata[i]); /*lint !e866*/
      consdata->vareventdata[i] = NULL;
   }

   SCIPfreeBlockMemoryArray(scip, &consdata->vareventdata, consdata->nvarexprs);
   consdata->vareventdata = NULL;

   return SCIP_OKAY;
}

/** processes variable fixing or bound change event */
static
SCIP_DECL_EVENTEXEC(processVarEvent)
{  /*lint --e{715}*/
   SCIP_EVENTTYPE eventtype;
   SCIP_CONSEXPR_EXPR* varexpr;
   SCIP_CONSDATA* consdata;
   SCIP_CONS* cons;
   SCIP_VAR* var;

   assert(eventdata != NULL);

   cons = ((SCIP_VAREVENTDATA*) eventdata)->cons;
   assert(cons != NULL);
   consdata = SCIPconsGetData(cons);
   assert(cons != NULL);

   varexpr = ((SCIP_VAREVENTDATA*) eventdata)->varexpr;
   assert(varexpr != NULL);
   assert(SCIPisConsExprExprVar(varexpr));

   var = SCIPgetConsExprExprVarVar(varexpr);
   assert(var != NULL);

   eventtype = SCIPeventGetType(event);
   assert((eventtype & SCIP_EVENTTYPE_BOUNDCHANGED) != 0 || (eventtype & SCIP_EVENTTYPE_VARFIXED) != 0);

   SCIPdebugMsg(scip, "  exec event %u for %s in %s\n", eventtype, SCIPvarGetName(var), SCIPconsGetName(cons));

   /* mark constraint to be propagated and simplified again */
   /* TODO: we only need to re-propagate if SCIP_EVENTTYPE_BOUNDTIGHTENED, but we need to reevaluate
    * the intervals (forward-propagation) when SCIP_EVENTTYPE_BOUNDRELAXED
    * at some point we should start using the intevaltag for this
    */
   if( (eventtype & SCIP_EVENTTYPE_BOUNDCHANGED) != (unsigned int) 0 )
   {
      SCIPdebugMsg(scip, "  propagate and simplify %s again\n", SCIPconsGetName(cons));
      consdata->ispropagated = FALSE;
      consdata->issimplified = FALSE;
   }
   if( (eventtype & SCIP_EVENTTYPE_VARFIXED) != (unsigned int) 0 )
   {
      consdata->issimplified = FALSE;
   }

   return SCIP_OKAY;
}

/** propagates variable locks through expression and adds lock to variables */
static
SCIP_RETCODE propagateLocks(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression */
   int                   nlockspos,          /**< number of positive locks */
   int                   nlocksneg           /**< number of negative locks */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSEXPRITERATOR_USERDATA ituserdata;

   assert(expr != NULL);

   /* if no locks, then nothing to do, then do nothing */
   if( nlockspos == 0 && nlocksneg == 0 )
      return SCIP_OKAY;

   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, expr, SCIP_CONSEXPRITERATOR_DFS, TRUE) );
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_ENTEREXPR | SCIP_CONSEXPRITERATOR_VISITINGCHILD | SCIP_CONSEXPRITERATOR_LEAVEEXPR);
   assert(SCIPexpriteratorGetCurrent(it) == expr); /* iterator should not have moved */

   /* store locks in root node */
   ituserdata.intvals[0] = nlockspos;
   ituserdata.intvals[1] = nlocksneg;
   SCIPexpriteratorSetCurrentUserData(it, ituserdata);

   while( !SCIPexpriteratorIsEnd(it) )
   {
      /* collect locks */
      ituserdata = SCIPexpriteratorGetCurrentUserData(it);
      nlockspos = ituserdata.intvals[0];
      nlocksneg = ituserdata.intvals[1];

      switch( SCIPexpriteratorGetStageDFS(it) )
      {
         case SCIP_CONSEXPRITERATOR_ENTEREXPR:
         {
            if( SCIPgetConsExprExprHdlr(expr) == SCIPgetConsExprExprHdlrVar(conshdlr) )
            {
               /* if a variable, then also add nlocksneg/nlockspos via SCIPaddVarLocks() */
               SCIP_CALL( SCIPaddVarLocks(scip, SCIPgetConsExprExprVarVar(expr), nlocksneg, nlockspos) );
            }

            /* add locks to expression */
            expr->nlockspos += nlockspos;
            expr->nlocksneg += nlocksneg;

            /* add monotonicity information if expression has been locked for the first time */
            if( expr->nlockspos == nlockspos && expr->nlocksneg == nlocksneg && expr->nchildren > 0
               && expr->exprhdlr->monotonicity != NULL )
            {
               int i;

               assert(expr->monotonicity == NULL);
               assert(expr->monotonicitysize == 0);

               SCIP_CALL( SCIPallocBlockMemoryArray(scip, &expr->monotonicity, expr->nchildren) );
               expr->monotonicitysize = expr->nchildren;

               /* store the monotonicity for each child */
               for( i = 0; i < expr->nchildren; ++i )
               {
                  SCIP_CALL( (*expr->exprhdlr->monotonicity)(scip, expr, i, &expr->monotonicity[i]) );
               }
            }
            break;
         }

         case SCIP_CONSEXPRITERATOR_LEAVEEXPR :
         {
            /* remove monotonicity information if expression has been unlocked */
            if( expr->nlockspos == 0 && expr->nlocksneg == 0 && expr->monotonicity != NULL )
            {
               assert(expr->monotonicitysize > 0);
               /* keep this assert for checking whether someone changed an expression without updating locks properly */
               assert(expr->monotonicitysize == expr->nchildren);

               SCIPfreeBlockMemoryArray(scip, &expr->monotonicity, expr->monotonicitysize);
               expr->monotonicitysize = 0;
            }
            break;
         }

         case SCIP_CONSEXPRITERATOR_VISITINGCHILD :
         {
            SCIP_MONOTONE monotonicity;

            assert(expr->monotonicity != NULL || expr->exprhdlr->monotonicity == NULL);

            /* get monotonicity of child */
            /* NOTE: the monotonicity stored in an expression might be different from the result obtained by
             * SCIPgetConsExprExprMonotonicity
             */
            monotonicity = expr->monotonicity != NULL ? expr->monotonicity[SCIPexpriteratorGetChildIdxDFS(it)] : SCIP_MONOTONE_UNKNOWN;

            /* compute resulting locks of the child expression */
            switch( monotonicity )
            {
               case SCIP_MONOTONE_INC:
                  ituserdata.intvals[0] = nlockspos;
                  ituserdata.intvals[1] = nlocksneg;
                  break;
               case SCIP_MONOTONE_DEC:
                  ituserdata.intvals[0] = nlocksneg;
                  ituserdata.intvals[1] = nlockspos;
                  break;
               case SCIP_MONOTONE_UNKNOWN:
                  ituserdata.intvals[0] = nlockspos + nlocksneg;
                  ituserdata.intvals[1] = nlockspos + nlocksneg;
                  break;
               case SCIP_MONOTONE_CONST:
                  ituserdata.intvals[0] = 0;
                  ituserdata.intvals[1] = 0;
                  break;
            }
            /* set locks in child expression */
            SCIPexpriteratorSetChildUserData(it, ituserdata);

            break;
         }

         default :
            /* you should never be here */
            SCIPABORT();
            break;
      }

      expr = SCIPexpriteratorGetNext(it);
   }

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** main function for adding locks to expressions and variables; locks for an expression constraint are used to update
 *  locks for all sub-expressions and variables; locks of expressions depend on the monotonicity of expressions
 *  w.r.t. their children, e.g., consider the constraint x^2 <= 1 with x in [-2,-1] implies an up-lock for the root
 *  expression (pow) and a down-lock for its child x because x^2 is decreasing on [-2,-1]; since the monotonicity (and thus
 *  the locks) might also depend on variable bounds, the function remembers the computed monotonicity information ofcan
 *  each expression until all locks of an expression have been removed, which implies that updating the monotonicity
 *  information during the next locking of this expression does not break existing locks
 *
 *  @note when modifying the structure of an expression, e.g., during simplification, it is necessary to remove all
 *        locks from an expression and repropagating them after the structural changes have been applied; because of
 *        existing common sub-expressions, it might be necessary to remove the locks of all constraints to ensure
 *        that an expression is unlocked (see canonicalizeConstraints() for an example)
 */
static
SCIP_RETCODE addLocks(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< expression constraint */
   int                   nlockspos,          /**< number of positive rounding locks */
   int                   nlocksneg           /**< number of negative rounding locks */
   )
{
   SCIP_CONSDATA* consdata;

   assert(cons != NULL);

   if( nlockspos == 0 && nlocksneg == 0 )
      return SCIP_OKAY;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* no constraint sides -> nothing to lock */
   if( SCIPisInfinity(scip, consdata->rhs) && SCIPisInfinity(scip, -consdata->lhs) )
      return SCIP_OKAY;

   /* call interval evaluation when root expression is locked for the first time */
   if( consdata->expr->nlockspos == 0 && consdata->expr->nlocksneg == 0 )
   {
      SCIP_CALL( SCIPevalConsExprExprInterval(scip, SCIPconsGetHdlr(cons), consdata->expr, 0, NULL, NULL) );
   }

   /* remember locks */
   consdata->nlockspos += nlockspos;
   consdata->nlocksneg += nlocksneg;

   assert(consdata->nlockspos >= 0);
   assert(consdata->nlocksneg >= 0);

   /* compute locks for lock propagation */
   if( !SCIPisInfinity(scip, consdata->rhs) && !SCIPisInfinity(scip, -consdata->lhs) )
   {
      SCIP_CALL( propagateLocks(scip, consdata->expr, nlockspos + nlocksneg, nlockspos + nlocksneg));
   }
   else if( !SCIPisInfinity(scip, consdata->rhs) )
   {
      SCIP_CALL( propagateLocks(scip, consdata->expr, nlockspos, nlocksneg));
   }
   else
   {
      assert(!SCIPisInfinity(scip, -consdata->lhs));
      SCIP_CALL( propagateLocks(scip, consdata->expr, nlocksneg, nlockspos));
   }

   return SCIP_OKAY;
}

/** returns an equivalent expression for a given expression if possible; it adds the expression to key2expr if the map
 *  does not contain the key
 */
static
SCIP_RETCODE findEqualExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR *  expr,               /**< expression to replace */
   SCIP_MULTIHASH*       key2expr,           /**< mapping of hashes to expressions */
   SCIP_CONSEXPR_EXPR**  newexpr             /**< pointer to store an equivalent expression (NULL if there is none) */
   )
{  /*lint --e{438}*/
   SCIP_MULTIHASHLIST* multihashlist;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(key2expr != NULL);
   assert(newexpr != NULL);

   *newexpr = NULL;
   multihashlist = NULL;

   do
   {
      /* search for an equivalent expression */
      *newexpr = (SCIP_CONSEXPR_EXPR*)(SCIPmultihashRetrieveNext(key2expr, &multihashlist, (void*)expr));

      if( *newexpr == NULL )
      {
         /* processed all expressions like expr from hash table, so insert expr */
         SCIP_CALL( SCIPmultihashInsert(key2expr, (void*) expr) );
         break;
      }
      else if( expr != *newexpr )
      {
         assert(SCIPcompareConsExprExprs(expr, *newexpr) == 0);
         break;
      }
      else
      {
         /* can not replace expr since it is already contained in the hashtablelist */
         assert(expr == *newexpr);
         *newexpr = NULL;
         break;
      }
   }
   while( TRUE ); /*lint !e506*/

   return SCIP_OKAY;
}

/** hashes an expression using an already existing iterator
 *
 * The iterator must by of type DFS with allowrevisit=FALSE and the only leaveexpr stage enabled.
 * The hashes of all visited expressions will be stored in the iterators expression data.
 */
static
SCIP_RETCODE hashExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression to hash */
   SCIP_CONSEXPR_ITERATOR* hashiterator,     /**< iterator to use for hashing */
   int*                  nvisitedexprs       /**< counter to increment by the number of expressions visited, or NULL */
   )
{
   SCIP_CONSEXPRITERATOR_USERDATA iterdata;
   unsigned int* childrenhashes;
   int childrenhashessize;
   int i;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(hashiterator != NULL);

   childrenhashessize = 5;
   SCIP_CALL( SCIPallocBufferArray(scip, &childrenhashes, childrenhashessize) );

   for( expr = SCIPexpriteratorRestartDFS(hashiterator, expr); !SCIPexpriteratorIsEnd(hashiterator); expr = SCIPexpriteratorGetNext(hashiterator) ) /*lint !e441*/
   {
      assert(SCIPexpriteratorGetStageDFS(hashiterator) == SCIP_CONSEXPRITERATOR_LEAVEEXPR);

      if( nvisitedexprs != NULL )
         ++*nvisitedexprs;

      /* collect hashes of children */
      if( childrenhashessize < expr->nchildren )
      {
         childrenhashessize = SCIPcalcMemGrowSize(scip, expr->nchildren);
         SCIP_CALL( SCIPreallocBufferArray(scip, &childrenhashes, childrenhashessize) );
      }
      for( i = 0; i < expr->nchildren; ++i )
         childrenhashes[i] = SCIPexpriteratorGetExprUserData(hashiterator, expr->children[i]).uintval;

      SCIP_CALL( SCIPhashConsExprExprHdlr(scip, expr, &iterdata.uintval, childrenhashes) );

      SCIPexpriteratorSetCurrentUserData(hashiterator, iterdata);
   }

   SCIPfreeBufferArray(scip, &childrenhashes);

   return SCIP_OKAY;
}

/** get key of hash element */
static
SCIP_DECL_HASHGETKEY(hashCommonSubexprGetKey)
{
   return elem;
}  /*lint !e715*/

/** checks if two expressions are structurally the same */
static
SCIP_DECL_HASHKEYEQ(hashCommonSubexprEq)
{
   SCIP_CONSEXPR_EXPR* expr1;
   SCIP_CONSEXPR_EXPR* expr2;

   expr1 = (SCIP_CONSEXPR_EXPR*)key1;
   expr2 = (SCIP_CONSEXPR_EXPR*)key2;
   assert(expr1 != NULL);
   assert(expr2 != NULL);

   return expr1 == expr2 || SCIPcompareConsExprExprs(expr1, expr2) == 0;
}  /*lint !e715*/

/** get value of hash element when comparing with another expression */
static
SCIP_DECL_HASHKEYVAL(hashCommonSubexprKeyval)
{
   SCIP_CONSEXPR_EXPR* expr;
   SCIP_CONSEXPR_ITERATOR* hashiterator;

   expr = (SCIP_CONSEXPR_EXPR*) key;
   assert(expr != NULL);

   hashiterator = (SCIP_CONSEXPR_ITERATOR*) userptr;
   assert(hashiterator != NULL);

   return SCIPexpriteratorGetExprUserData(hashiterator, expr).uintval;
}  /*lint !e715*/

/* export this function here, so it can be used by unittests but is not really part of the API */
/** replaces common sub-expressions in the current expression graph by using a hash key for each expression; the
 *  algorithm consists of two steps:
 *
 *  1. traverse through all expressions trees of given constraints and compute for each of them a (not necessarily
 *     unique) hash
 *
 *  2. initialize an empty hash table and traverse through all expression; check for each of them if we can find a
 *     structural equivalent expression in the hash table; if yes we replace the expression by the expression inside the
 *     hash table, otherwise we add it to the hash table
 *
 *  @note the hash keys of the expressions are used for the hashing inside the hash table; to compute if two expressions
 *  (with the same hash) are structurally the same we use the function SCIPcompareConsExprExprs()
 */
static
SCIP_RETCODE replaceCommonSubexpressions(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           conss,              /**< constraints */
   int                   nconss              /**< total number of constraints */
   )
{
   SCIP_CONSEXPR_ITERATOR* hashiterator;
   SCIP_CONSEXPR_ITERATOR* repliterator;
   SCIP_MULTIHASH* key2expr;
   SCIP_CONSDATA* consdata;
   int i;
   int nexprs = 0;

   assert(scip != NULL);
   assert(conss != NULL);
   assert(nconss >= 0);

   if( nconss == 0 )
      return SCIP_OKAY;

   SCIP_CALL( SCIPexpriteratorCreate(&hashiterator, SCIPconsGetHdlr(conss[0]), SCIPblkmem(scip)) );

   /* compute all hashes for each sub-expression */
   for( i = 0; i < nconss; ++i )
   {
      assert(conss[i] != NULL);

      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      if( consdata->expr == NULL )
         continue;

      if( !SCIPexpriteratorIsInit(hashiterator) )
      {
         /* first constraint with non-NULL expr: initialize iterator (set type and stopstage) */
         SCIP_CALL( SCIPexpriteratorInit(hashiterator, consdata->expr, SCIP_CONSEXPRITERATOR_DFS, FALSE) );
         SCIPexpriteratorSetStagesDFS(hashiterator, SCIP_CONSEXPRITERATOR_LEAVEEXPR);
      }

      SCIP_CALL( hashExpr(scip, consdata->expr, hashiterator, &nexprs) );
   }

   /* replace equivalent sub-expressions */
   SCIP_CALL( SCIPmultihashCreate(&key2expr, SCIPblkmem(scip), nexprs,
         hashCommonSubexprGetKey, hashCommonSubexprEq, hashCommonSubexprKeyval, (void*)hashiterator) );

   SCIP_CALL( SCIPexpriteratorCreate(&repliterator, SCIPconsGetHdlr(conss[0]), SCIPblkmem(scip)) );

   for( i = 0; i < nconss; ++i )
   {
      SCIP_CONSEXPR_EXPR* newroot;
      SCIP_CONSEXPR_EXPR* newchild;
      SCIP_CONSEXPR_EXPR* child;

      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      if( consdata->expr == NULL )
         continue;

      /* check the root for equivalence separately first */
      SCIP_CALL( findEqualExpr(scip, consdata->expr, key2expr, &newroot) );

      if( newroot != NULL )
      {
         assert(newroot != consdata->expr);
         assert(SCIPcompareConsExprExprs(consdata->expr, newroot) == 0);

         SCIPdebugMsg(scip, "replacing common root expression of constraint <%s>: %p -> %p\n", SCIPconsGetName(conss[i]), (void*)consdata->expr, (void*)newroot);

         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &consdata->expr) );

         consdata->expr = newroot;
         SCIPcaptureConsExprExpr(newroot);

         continue;
      }

      /* replace equivalent sub-expressions in the tree */
      SCIP_CALL( SCIPexpriteratorInit(repliterator, consdata->expr, SCIP_CONSEXPRITERATOR_DFS, FALSE) );
      SCIPexpriteratorSetStagesDFS(repliterator, SCIP_CONSEXPRITERATOR_VISITINGCHILD);

      while( !SCIPexpriteratorIsEnd(repliterator) )
      {
         child = SCIPexpriteratorGetChildExprDFS(repliterator);
         assert(child != NULL);

         /* try to find an equivalent expression */
         SCIP_CALL( findEqualExpr(scip, child, key2expr, &newchild) );

         /* replace child with newchild */
         if( newchild != NULL )
         {
            assert(child != newchild);
            assert(SCIPcompareConsExprExprs(child, newchild) == 0);

            SCIPdebugMsg(scip, "replacing common child expression %p -> %p\n", (void*)child, (void*)newchild);

            SCIP_CALL( SCIPreplaceConsExprExprChild(scip, SCIPexpriteratorGetCurrent(repliterator), SCIPexpriteratorGetChildIdxDFS(repliterator), newchild) );

            (void) SCIPexpriteratorSkipDFS(repliterator);
         }
         else
         {
            (void) SCIPexpriteratorGetNext(repliterator);
         }
      }
   }

   /* free memory */
   SCIPexpriteratorFree(&repliterator);
   SCIPmultihashFree(&key2expr);
   SCIPexpriteratorFree(&hashiterator);

   return SCIP_OKAY;
}

/** helper function to either simplify or reformulate an expression and its subexpressions */
static
SCIP_RETCODE reformulateConsExprExpr(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          conshdlr,         /**< constraint handler */
   SCIP_CONSEXPR_EXPR*     rootexpr,         /**< expression to be simplified */
   SCIP_Bool               simplify,         /**< should the expression be simplified or reformulated? */
   SCIP_CONSEXPR_EXPR**    simplified,       /**< buffer to store simplified expression */
   SCIP_Bool*              changed           /**< buffer to store if rootexpr actually changed */
   )
{
   SCIP_CONSEXPR_EXPR* expr;
   SCIP_CONSEXPR_ITERATOR* it;

   assert(scip != NULL);
   assert(rootexpr != NULL);
   assert(simplified != NULL);

   /* simplify bottom up
    * when leaving an expression it simplifies it and stores the simplified expr in its iterators expression data
    * after the child was visited, it is replaced with the simplified expr
    */
   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, rootexpr, SCIP_CONSEXPRITERATOR_DFS, TRUE) );  /* TODO can we set allowrevisited to FALSE?*/
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_VISITEDCHILD | SCIP_CONSEXPRITERATOR_LEAVEEXPR);

   *changed = FALSE;
   for( expr = SCIPexpriteratorGetCurrent(it); !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
   {
      switch( SCIPexpriteratorGetStageDFS(it) )
      {
         case SCIP_CONSEXPRITERATOR_VISITEDCHILD:
         {
            SCIP_CONSEXPR_EXPR* newchild;
            SCIP_CONSEXPR_EXPR* child;

            newchild = (SCIP_CONSEXPR_EXPR*)SCIPexpriteratorGetChildUserDataDFS(it).ptrval;
            child = SCIPexpriteratorGetChildExprDFS(it);
            assert(newchild != NULL);

            /* if child got simplified, replace it with the new child */
            if( newchild != child )
            {
               SCIP_CALL( SCIPreplaceConsExprExprChild(scip, expr, SCIPexpriteratorGetChildIdxDFS(it), newchild) );
            }

            /* we do not need to hold newchild anymore */
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &newchild) );

            break;
         }

         case SCIP_CONSEXPRITERATOR_LEAVEEXPR:
         {
            SCIP_CONSEXPR_EXPR* refexpr = NULL;
            SCIP_CONSEXPRITERATOR_USERDATA iterdata;

            /* use simplification of expression handlers */
            if( simplify )
            {
               if( SCIPhasConsExprExprHdlrSimplify(expr->exprhdlr) )
               {
                  SCIP_CALL( SCIPsimplifyConsExprExprHdlr(scip, expr, &refexpr) );
                  if( expr != refexpr )
                     *changed = TRUE;
               }
               else
               {
                  assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), "sum")  != 0);
                  assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), "prod") != 0);
                  assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), "var") != 0);
                  assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), "abs") != 0);
                  assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), "log") != 0);
                  assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), "exp") != 0);
                  assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), "pow") != 0);
                  assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), "sin") != 0);
                  assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), "cos") != 0);

                  /* if an expression handler doesn't implement simplify, we assume all those type of expressions are simplified
                   * we have to capture it, since it must simulate a "normal" simplified call in which a new expression is created
                   */
                  refexpr = expr;
                  SCIPcaptureConsExprExpr(refexpr);
               }
               assert(refexpr != NULL);
            }
            else /* use nonlinear handler to reformulate the expression */
            {
               SCIP_CONSHDLRDATA* conshdlrdata;
               int k;

               conshdlrdata = SCIPconshdlrGetData(conshdlr);
               assert(conshdlrdata != NULL);

               /* iterate through nonlinear handlers and call reformulation callbacks;
                *
                * TODO store nonlinear handlers that implement the reformulation callback separately
                * TODO sort nonlinear handlers according to their priorities
                */
               for( k = 0; k < conshdlrdata->nnlhdlrs; ++k )
               {
                  assert(conshdlrdata->nlhdlrs[k] != NULL);

                  if( SCIPhasConsExprNlhdlrReformulate(conshdlrdata->nlhdlrs[k]) )
                  {
                     SCIP_CALL( SCIPreformulateConsExprNlhdlr(scip, conshdlr, conshdlrdata->nlhdlrs[k], expr, &refexpr) );
                     assert(refexpr != NULL);

                     /* stop calling other nonlinear handlers as soon as the reformulation was successful */
                     if( refexpr != expr )
                     {
                        *changed = TRUE;
                        break;
                     }
                  }
               }

               /* no nonlinear handlers implements the reformulation callback -> capture expression manually */
               if( refexpr == NULL )
               {
                  refexpr = expr;
                  SCIPcaptureConsExprExpr(refexpr);
               }
            }

            iterdata.ptrval = (void*) refexpr;
            SCIPexpriteratorSetCurrentUserData(it, iterdata);

            break;
         }

         default:
            SCIPABORT(); /* we should never be called in this stage */
            break;
      }
   }

   *simplified = (SCIP_CONSEXPR_EXPR*)SCIPexpriteratorGetExprUserData(it, rootexpr).ptrval;
   assert(*simplified != NULL);

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}


/** scales the sides of the constraint l <= sum_i c_i f_i(x) <= r according to the following rules:
 *
 *  let n_+ the number of positive coefficients c_i and n_- be the number of negative coefficients
 *
 *   i. scale by -1 if n_+ < n_-
 *
 *  ii. scale by -1 if n_+ = n_- & r = INF
 */
static
SCIP_RETCODE scaleConsSides(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_CONS*            cons,               /**< expression constraint */
   SCIP_Bool*            changed             /**< buffer to store if the expression of cons changed */
   )
{
   SCIP_CONSDATA* consdata;
   int i;

   assert(cons != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   if( SCIPgetConsExprExprHdlr(consdata->expr) == SCIPgetConsExprExprHdlrSum(conshdlr) )
   {
      SCIP_Real* coefs;
      SCIP_Real constant;
      int nchildren;
      int counter = 0;

      coefs = SCIPgetConsExprExprSumCoefs(consdata->expr);
      constant = SCIPgetConsExprExprSumConstant(consdata->expr);
      nchildren = SCIPgetConsExprExprNChildren(consdata->expr);

      /* handle special case when constraint is l <= -f(x) <= r and f(x) not a sum: simplfy ensures f is not a sum */
      if( nchildren == 1 && constant == 0.0 && coefs[0] == -1.0 )
      {
         SCIP_CONSEXPR_EXPR* expr;
         expr = consdata->expr;

         consdata->expr = SCIPgetConsExprExprChildren(expr)[0];
         assert(SCIPgetConsExprExprHdlr(consdata->expr) != SCIPgetConsExprExprHdlrSum(conshdlr));

         SCIPcaptureConsExprExpr(consdata->expr);

         SCIPswapReals(&consdata->lhs, &consdata->rhs);
         consdata->lhs = -consdata->lhs;
         consdata->rhs = -consdata->rhs;

         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &expr) );
         *changed = TRUE;
         return SCIP_OKAY;
      }

      /* compute n_+ - n_i */
      for( i = 0; i < nchildren; ++i )
         counter += coefs[i] > 0 ? 1 : -1;

      if( counter < 0 || (counter == 0 && SCIPisInfinity(scip, consdata->rhs)) )
      {
         SCIP_CONSEXPR_EXPR* expr;
         SCIP_Real* newcoefs;

         /* allocate memory */
         SCIP_CALL( SCIPallocBufferArray(scip, &newcoefs, nchildren) );

         for( i = 0; i < nchildren; ++i )
            newcoefs[i] = -coefs[i];

         /* create a new sum expression */
         SCIP_CALL( SCIPcreateConsExprExprSum(scip, conshdlr, &expr, nchildren, SCIPgetConsExprExprChildren(consdata->expr), newcoefs, -constant) );

         /* replace expression in constraint data and scale sides */
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &consdata->expr) );
         consdata->expr = expr;
         SCIPswapReals(&consdata->lhs, &consdata->rhs);
         consdata->lhs = -consdata->lhs;
         consdata->rhs = -consdata->rhs;

         /* free memory */
         SCIPfreeBufferArray(scip, &newcoefs);

         *changed = TRUE;
      }
   }

   return SCIP_OKAY;
}

/** simplifies expressions and replaces common subexpressions for a set of constraints
 * @todo put the constant to the constraint sides
 */
static
SCIP_RETCODE canonicalizeConstraints(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_CONS**           conss,              /**< constraints */
   int                   nconss,             /**< total number of constraints */
   SCIP_Bool*            infeasible          /**< buffer to store whether infeasibility has been detected */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP_CONSEXPR_ITERATOR* it;
   int* nlockspos;
   int* nlocksneg;
   SCIP_Bool havechange;
   SCIP_Bool reformulate = FALSE;
   int i;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(conss != NULL);
   assert(nconss >= 0);
   assert(infeasible != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIP_CALL( SCIPstartClock(scip, conshdlrdata->canonicalizetime) );

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );

   *infeasible = FALSE;

   /* check whether at least one nonlinear handler implements the reformulation callback */
   for( i = 0; i < conshdlrdata->nnlhdlrs; ++i )
   {
      if( SCIPhasConsExprNlhdlrReformulate(conshdlrdata->nlhdlrs[i]) )
      {
         reformulate = TRUE;
         break;
      }
   }

   havechange = FALSE;

   /* free nonlinear handlers information from expressions */  /* TODO can skip this in first presolve round */
   SCIP_CALL( SCIPexpriteratorInit(it, NULL, SCIP_CONSEXPRITERATOR_DFS, FALSE) );
   for( i = 0; i < nconss; ++i )
   {
      SCIP_CONSEXPR_EXPR* expr;

      assert(conss != NULL);
      assert(conss[i] != NULL);

      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      for( expr = SCIPexpriteratorRestartDFS(it, consdata->expr); !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
      {
         SCIPdebugMsg(scip, "free nonlinear handler data for expression %p\n", (void*)expr);

         assert(expr->auxvar == NULL);  /* should not have been created yet or have been removed in INITPRE (if restart) */

         /* remove nonlinear handlers in expression and their data */
         SCIP_CALL( freeEnfoData(scip, expr, FALSE) );
      }
   }

   /* allocate memory for storing locks of each constraint */
   SCIP_CALL( SCIPallocBufferArray(scip, &nlockspos, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nlocksneg, nconss) );

   /* unlock all constraints */
   for( i = 0; i < nconss; ++i )
   {
      assert(conss[i] != NULL);

      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      /* remember locks */
      nlockspos[i] = consdata->nlockspos;
      nlocksneg[i] = consdata->nlocksneg;

      /* remove locks */
      SCIP_CALL( addLocks(scip, conss[i], -consdata->nlockspos, -consdata->nlocksneg) );
      assert(consdata->nlockspos == 0);
      assert(consdata->nlocksneg == 0);
   }

#ifndef NDEBUG
   /* check whether all locks of each expression have been removed */
   for( i = 0; i < nconss; ++i )
   {
      SCIP_CONSEXPR_EXPR* expr;

      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      SCIP_CALL( SCIPexpriteratorInit(it, consdata->expr, SCIP_CONSEXPRITERATOR_RTOPOLOGIC, TRUE) );
      for( expr = consdata->expr; !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
      {
         assert(expr != NULL);
         assert(expr->nlocksneg == 0);
         assert(expr->nlockspos == 0);
      }
   }
#endif

   for( i = 0; i < nconss; ++i )
   {
      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      /* call simplify for each expression */
      if( !consdata->issimplified && consdata->expr != NULL )
      {
         SCIP_CONSEXPR_EXPR* simplified;
         SCIP_Bool changed;

         changed = FALSE;
         SCIP_CALL( SCIPsimplifyConsExprExpr(scip, conshdlr, consdata->expr, &simplified, &changed) );
         consdata->issimplified = TRUE;

         if( changed )
            havechange = TRUE;

         /* If root expression changed, then we need to take care updating the locks as well (the consdata is the one holding consdata->expr "as a child").
          * If root expression did not change, some subexpression may still have changed, but the locks were taking care of in the corresponding SCIPreplaceConsExprExprChild() call.
          */
         if( simplified != consdata->expr )
         {
            assert(changed);

            /* release old expression */
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &consdata->expr) );

            /* store simplified expression */
            consdata->expr = simplified;
         }
         else
         {
            /* The simplify captures simplified in any case, also if nothing has changed.
             * Therefore, we have to release it here.
             */
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &simplified) );
         }

         /* scale constraint sides */
         SCIP_CALL( scaleConsSides(scip, conshdlr, conss[i], &changed) );
      }

      /* call reformulation callback of nonlinear handlers for each expression */
      if( reformulate && SCIPgetStage(scip) == SCIP_STAGE_PRESOLVING )
      {
         SCIP_CONSEXPR_EXPR* refexpr;
         SCIP_Bool changed;

         if( consdata->expr != NULL )
         {
            SCIP_CALL( SCIPreformulateConsExprExpr(scip, conshdlr, consdata->expr, &refexpr, &changed) );

            if( changed )
               havechange = TRUE;

            if( refexpr != consdata->expr )
            {
               assert(changed);

               /* release old expression */
               SCIP_CALL( SCIPreleaseConsExprExpr(scip, &consdata->expr) );

               /* store simplified expression */
               consdata->expr = refexpr;
            }
            else
            {
               /* The reformulation captures simplified in any case, also if nothing has changed.
                * Therefore, we have to release it here.
                */
               SCIP_CALL( SCIPreleaseConsExprExpr(scip, &refexpr) );
            }
         }
      }
   }

   /* replace common subexpressions */
   if( havechange )
   {
      SCIP_CALL( replaceCommonSubexpressions(scip, conss, nconss) );

      /* FIXME: this is a dirty hack for updating the variable expressions stored inside an expression which might have
       * been changed after simplification; now we completely recollect all variable expression and variable events
       */
      for( i = 0; i < nconss; ++i )
      {
         SCIP_CALL( dropVarEvents(scip, conshdlrdata->eventhdlr, conss[i]) );
         SCIP_CALL( freeVarExprs(scip, SCIPconsGetData(conss[i])) );
      }
      for( i = 0; i < nconss; ++i )
      {
         SCIP_CALL( storeVarExprs(scip, conshdlr, SCIPconsGetData(conss[i])) );
         SCIP_CALL( catchVarEvents(scip, conshdlrdata->eventhdlr, conss[i]) );
      }
   }

   /* restore locks */
   for( i = 0; i < nconss; ++i )
   {
      SCIP_CALL( addLocks(scip, conss[i], nlockspos[i], nlocksneg[i]) );
   }

   /* run nlhdlr detect if in presolving stage (that is, not in exitpre) */
   if( SCIPgetStage(scip) == SCIP_STAGE_PRESOLVING )
   {
      SCIP_CALL( detectNlhdlrs(scip, conshdlr, conss, nconss, infeasible) );
   }

   /* free allocated memory */
   SCIPfreeBufferArray(scip, &nlocksneg);
   SCIPfreeBufferArray(scip, &nlockspos);
   SCIPexpriteratorFree(&it);

   SCIP_CALL( SCIPstopClock(scip, conshdlrdata->canonicalizetime) );

   return SCIP_OKAY;
}

/** @name Parsing methods
 * @{
 * Here is an attempt at defining the grammar of an expression.
 * We use upper case names for variables (in the grammar sense) and terminals are between "".
 * Loosely speaking, a Base will be any "block", a Factor is a Base to a power, a Term is a product of Factors
 * and an Expression is a sum of terms.
 * The actual definition:
 * <pre>
 * Expression -> ["+" | "-"] Term { ("+" | "-" | "number *") ] Term }
 * Term       -> Factor { ("*" | "/" ) Factor }
 * Factor     -> Base [ "^" "number" | "^(" "number" ")" ]
 * Base       -> "number" | "<varname>" | "(" Expression ")" | Op "(" OpExpression ")
 * </pre>
 * where [a|b] means a or b or none, (a|b) means a or b, {a} means 0 or more a.
 *
 * Note that Op and OpExpression are undefined. Op corresponds to the name of an expression handler and
 * OpExpression to whatever string the expression handler accepts (through its parse method).
 *
 * parse(Expr|Term|Base) returns an SCIP_CONSEXPR_EXPR
 *
 * @todo We can change the grammar so that Factor becomes base and we allow a Term to be
 *       <pre> Term       -> Factor { ("*" | "/" | "^") Factor } </pre>
 */

#ifdef PARSE_DEBUG
#define debugParse                      printf
#else
#define debugParse                      while( FALSE ) printf
#endif
static
SCIP_RETCODE parseExpr(SCIP*, SCIP_CONSHDLR*, SCIP_HASHMAP*, const char*, const char**, SCIP_CONSEXPR_EXPR**);

/** Parses base to build a value, variable, sum, or function-like ("func(...)") expression.
 * <pre>
 * Base       -> "number" | "<varname>" | "(" Expression ")" | Op "(" OpExpression ")
 * </pre>
 */
static
SCIP_RETCODE parseBase(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_HASHMAP*         vartoexprvarmap,    /**< hashmap to map between SCIP vars and var expressions */
   const char*           expr,               /**< expr that we are parsing */
   const char**          newpos,             /**< buffer to store the position of expr where we finished reading */
   SCIP_CONSEXPR_EXPR**  basetree            /**< buffer to store the expr parsed by Base */
   )
{
   SCIP_VAR* var;

   debugParse("parsing base from %s\n", expr); /*lint !e506 !e681*/

   /* ignore whitespace */
   while( isspace((unsigned char)*expr) )
      ++expr;

   if( *expr == '\0' )
   {
      SCIPerrorMessage("Unexpected end of expression string\n");
      return SCIP_READERROR;
   }

   if( *expr == '<' )
   {
      /* parse a variable */
      SCIP_CALL( SCIPparseVarName(scip, expr, &var, (char**)newpos) );

      if( var == NULL )
      {
         SCIPerrorMessage("Could not find variable with name '%s'\n", expr);
         return SCIP_READERROR;
      }
      expr = *newpos;

      /* check if we have already created an expression out of this var */
      if( SCIPhashmapExists(vartoexprvarmap, (void *)var) )
      {
         debugParse("Variable %s has been parsed, capturing its expression\n", SCIPvarGetName(var)); /*lint !e506 !e681*/
         *basetree = (SCIP_CONSEXPR_EXPR*)SCIPhashmapGetImage(vartoexprvarmap, (void *)var);
         SCIPcaptureConsExprExpr(*basetree);
      }
      else
      {
         debugParse("First time parsing variable %s, creating varexpr and adding it to hashmap\n", SCIPvarGetName(var)); /*lint !e506 !e681*/
         SCIP_CALL( SCIPcreateConsExprExprVar(scip, conshdlr, basetree, var) );
         SCIP_CALL( SCIPhashmapInsert(vartoexprvarmap, (void*)var, (void*)(*basetree)) );
      }
   }
   else if( *expr == '(' )
   {
      /* parse expression */
      SCIP_CALL( parseExpr(scip, conshdlr, vartoexprvarmap, ++expr, newpos, basetree) );
      expr = *newpos;

      /* expect ')' */
      if( *expr != ')' )
      {
         SCIPerrorMessage("Read a '(', parsed expression inside --> expecting closing ')'. Got <%c>: rest of string <%s>\n", *expr, expr);
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, basetree) );
         return SCIP_READERROR;
      }
      ++expr;
      debugParse("Done parsing expression, continue with <%s>\n", expr); /*lint !e506 !e681*/
   }
   else if( isdigit(*expr) )
   {
      /* parse number */
      SCIP_Real value;
      if( !SCIPstrToRealValue(expr, &value, (char**)&expr) )
      {
         SCIPerrorMessage("error parsing number from <%s>\n", expr);
         return SCIP_READERROR;
      }
      debugParse("Parsed value %g, creating a value-expression.\n", value); /*lint !e506 !e681*/
      SCIP_CALL( SCIPcreateConsExprExprValue(scip, conshdlr, basetree, value) );
   }
   else if( isalpha(*expr) )
   {
      /* a (function) name is coming, should find exprhandler with such name */
      int i;
      char operatorname[SCIP_MAXSTRLEN];
      SCIP_CONSEXPR_EXPRHDLR* exprhdlr;
      SCIP_Bool success;

      /* get name */
      i = 0;
      while( *expr != '(' && !isspace((unsigned char)*expr) && *expr != '\0' )
      {
         operatorname[i] = *expr;
         ++expr;
         ++i;
      }
      operatorname[i] = '\0';

      /* after name we must see a '(' */
      if( *expr != '(' )
      {
         SCIPerrorMessage("Expected '(' after operator name <%s>, but got %s.\n", operatorname, expr);
         return SCIP_READERROR;
      }

      /* search for expression handler */
      exprhdlr = SCIPfindConsExprExprHdlr(conshdlr, operatorname);

      /* check expression handler exists and has a parsing method */
      if( exprhdlr == NULL )
      {
         SCIPerrorMessage("No expression handler with name <%s> found.\n", operatorname);
         return SCIP_READERROR;
      }

      ++expr;
      SCIP_CALL( SCIPparseConsExprExprHdlr(scip, conshdlr, exprhdlr, expr, newpos, basetree, &success) );

      if( !success )
      {
         SCIPerrorMessage("Error while expression handler <%s> was parsing %s\n", operatorname, expr);
         assert(*basetree == NULL);
         return SCIP_READERROR;
      }
      expr = *newpos;

      /* we should see the ')' of Op "(" OpExpression ") */
      assert(*expr == ')');

      /* move one character forward */
      ++expr;
   }
   else
   {
      /* Base -> "number" | "<varname>" | "(" Expression ")" | Op "(" OpExpression ") */
      SCIPerrorMessage("Expected a number, (expression), <varname>, Opname(Opexpr), instead got <%c> from %s\n", *expr, expr);
      return SCIP_READERROR;
   }

   *newpos = expr;

   return SCIP_OKAY;
}

/** Parses a factor and builds a product-expression if there is an exponent, otherwise returns the base expression.
 * <pre>
 * Factor -> Base [ "^" "number" | "^(" "number" ")" ]
 * </pre>
 */
static
SCIP_RETCODE parseFactor(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_Bool             isdenominator,      /**< whether factor is in the denominator */
   SCIP_HASHMAP*         vartoexprvarmap,    /**< hashmap to map between scip vars and var expressions */
   const char*           expr,               /**< expr that we are parsing */
   const char**          newpos,             /**< buffer to store the position of expr where we finished reading */
   SCIP_CONSEXPR_EXPR**  factortree          /**< buffer to store the expr parsed by Factor */
   )
{
   SCIP_CONSEXPR_EXPR*  basetree;
   SCIP_Real exponent;

   debugParse("parsing factor from %s\n", expr); /*lint !e506 !e681*/

   if( *expr == '\0' )
   {
      SCIPerrorMessage("Unexpected end of expression string.\n");
      return SCIP_READERROR;
   }

   /* parse Base */
   /* ignore whitespace */
   while( isspace((unsigned char)*expr) )
      ++expr;

   SCIP_CALL( parseBase(scip, conshdlr, vartoexprvarmap, expr, newpos, &basetree) );
   expr = *newpos;

   /* check if there is an exponent */
   /* ignore whitespace */
   while( isspace((unsigned char)*expr) )
      ++expr;
   if( *expr == '^' )
   {

      ++expr;
      while( isspace((unsigned char)*expr) )
         ++expr;

      if( *expr == '\0' )
      {
         SCIPerrorMessage("Unexpected end of expression string after '^'.\n");
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &basetree) );
         return SCIP_READERROR;
      }

      if( *expr == '(' )
      {
         ++expr;

         /* it is exponent with parenthesis; expect number possibly starting with + or - */
         if( !SCIPstrToRealValue(expr, &exponent, (char**)&expr) )
         {
            SCIPerrorMessage("error parsing number from <%s>\n", expr);
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &basetree) );
            return SCIP_READERROR;
         }

         /* expect the ')' */
         while( isspace((unsigned char)*expr) )
            ++expr;
         if( *expr != ')' )
         {
            SCIPerrorMessage("error in parsing exponent: expected ')', received <%c> from <%s>\n", *expr,  expr);
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &basetree) );
            return SCIP_READERROR;
         }
         ++expr;
      }
      else
      {
         /* no parenthesis, we should see just a positive number */

         /* expect a digit */
         if( isdigit(*expr) )
         {
            if( !SCIPstrToRealValue(expr, &exponent, (char**)&expr) )
            {
               SCIPerrorMessage("error parsing number from <%s>\n", expr);
               SCIP_CALL( SCIPreleaseConsExprExpr(scip, &basetree) );
               return SCIP_READERROR;
            }
         }
         else
         {
            SCIPerrorMessage("error in parsing exponent, expected a digit, received <%c> from <%s>\n", *expr,  expr);
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &basetree) );
            return SCIP_READERROR;
         }
      }

      debugParse("parsed the exponent %g\n", exponent); /*lint !e506 !e681*/
   }
   else
   {
      /* there is no explicit exponent */
      exponent = 1.0;
   }
   *newpos = expr;

   /* multiply with -1 when we are in the denominator */
   if( isdenominator )
      exponent *= -1.0;

   /* create power */
   if( exponent != 1.0 )
   {
      SCIP_CALL( SCIPcreateConsExprExprPow(scip, conshdlr, factortree, basetree, exponent) );
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &basetree) );
   }
   else
      /* Factor consists of this unique Base */
      *factortree = basetree;

   return SCIP_OKAY;
}

/** Parses a term and builds a product-expression, where each factor is a child.
 * <pre>
 * Term -> Factor { ("*" | "/" ) Factor }
 * </pre>
 */
static
SCIP_RETCODE parseTerm(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_HASHMAP*         vartoexprvarmap,    /**< hashmap to map between scip vars and var expressions */
   const char*           expr,               /**< expr that we are parsing */
   const char**          newpos,             /**< buffer to store the position of expr where we finished reading */
   SCIP_CONSEXPR_EXPR**  termtree            /**< buffer to store the expr parsed by Term */
   )
{
   SCIP_CONSEXPR_EXPR* factortree;

   debugParse("parsing term from %s\n", expr); /*lint !e506 !e681*/

   /* parse Factor */
   /* ignore whitespace */
   while( isspace((unsigned char)*expr) )
      ++expr;

   SCIP_CALL( parseFactor(scip, conshdlr, FALSE, vartoexprvarmap, expr, newpos, &factortree) );
   expr = *newpos;

   debugParse("back to parsing Term, continue parsing from %s\n", expr); /*lint !e506 !e681*/

   /* check if Terms has another Factor incoming */
   while( isspace((unsigned char)*expr) )
      ++expr;
   if( *expr == '*' || *expr == '/' )
   {
      /* initialize termtree as a product expression with a single term, so we can append the extra Factors */
      SCIP_CALL( SCIPcreateConsExprExprProduct(scip, conshdlr, termtree, 1, &factortree, 1.0) );
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &factortree) );

      /* loop: parse Factor, find next symbol */
      do
      {
         SCIP_RETCODE retcode;
         SCIP_Bool isdivision;

         isdivision = (*expr == '/') ? TRUE : FALSE;

         debugParse("while parsing term, read char %c\n", *expr); /*lint !e506 !e681*/

         ++expr;
         retcode = parseFactor(scip, conshdlr, isdivision, vartoexprvarmap, expr, newpos, &factortree);

         /* release termtree, if parseFactor fails with a read-error */
         if( retcode == SCIP_READERROR )
         {
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, termtree) );
         }
         SCIP_CALL( retcode );

         /* append newly created factor */
         SCIP_CALL( SCIPappendConsExprExprProductExpr(scip, *termtree, factortree) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &factortree) );

         /* find next symbol */
         expr = *newpos;
         while( isspace((unsigned char)*expr) )
            ++expr;
      } while( *expr == '*' || *expr == '/' );
   }
   else
   {
      /* Term consists of this unique factor */
      *termtree = factortree;
   }

   *newpos = expr;

   return SCIP_OKAY;
}

/** Parses an expression and builds a sum-expression with children.
 * <pre>
 * Expression -> ["+" | "-"] Term { ("+" | "-" | "number *") ] Term }
 * </pre>
 */
static
SCIP_RETCODE parseExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_HASHMAP*         vartoexprvarmap,    /**< hashmap to map between scip vars and var expressions */
   const char*           expr,               /**< expr that we are parsing */
   const char**          newpos,             /**< buffer to store the position of expr where we finished reading */
   SCIP_CONSEXPR_EXPR**  exprtree            /**< buffer to store the expr parsed by Expr */
   )
{
   SCIP_Real sign;
   SCIP_CONSEXPR_EXPR* termtree;

   debugParse("parsing expression %s\n", expr); /*lint !e506 !e681*/

   /* ignore whitespace */
   while( isspace((unsigned char)*expr) )
      ++expr;

   /* if '+' or '-', store it */
   sign = 1.0;
   if( *expr == '+' || *expr == '-' )
   {
      debugParse("while parsing expression, read char %c\n", *expr); /*lint !e506 !e681*/
      sign = *expr == '+' ? 1.0 : -1.0;
      ++expr;
   }

   SCIP_CALL( parseTerm(scip, conshdlr, vartoexprvarmap, expr, newpos, &termtree) );
   expr = *newpos;

   debugParse("back to parsing expression (we have the following term), continue parsing from %s\n", expr); /*lint !e506 !e681*/

   /* check if Expr has another Term incoming */
   while( isspace((unsigned char)*expr) )
      ++expr;
   if( *expr == '+' || *expr == '-' )
   {
      if( SCIPgetConsExprExprHdlr(termtree) == SCIPgetConsExprExprHdlrValue(conshdlr) )
      {
         /* initialize exprtree as a sum expression with a constant only, so we can append the following terms */
         SCIP_CALL( SCIPcreateConsExprExprSum(scip, conshdlr, exprtree, 0, NULL, NULL, sign * SCIPgetConsExprExprValueValue(termtree)) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &termtree) );
      }
      else
      {
         /* initialize exprtree as a sum expression with a single term, so we can append the following terms */
         SCIP_CALL( SCIPcreateConsExprExprSum(scip, conshdlr, exprtree, 1, &termtree, &sign, 0.0) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &termtree) );
      }

      /* loop: parse Term, find next symbol */
      do
      {
         SCIP_RETCODE retcode;
         SCIP_Real coef;

         /* check if we have a "coef * <term>" */
         if( SCIPstrToRealValue(expr, &coef, (char**)newpos) )
         {
            while( isspace((unsigned char)**newpos) )
               ++(*newpos);

            if( **newpos != '*' )
            {
               /* no '*', so fall back to parsing term after sign */
               coef = (*expr == '+') ? 1.0 : -1.0;
               ++expr;
            }
            else
            {
               /* keep coefficient in coef and continue parsing term after coefficient */
               expr = (*newpos)+1;

               while( isspace((unsigned char)*expr) )
                  ++expr;
            }
         }
         else
         {
            coef = (*expr == '+') ? 1.0 : -1.0;
            ++expr;
         }

         debugParse("while parsing expression, read coefficient %g\n", coef); /*lint !e506 !e681*/

         retcode = parseTerm(scip, conshdlr, vartoexprvarmap, expr, newpos, &termtree);

         /* release exprtree if parseTerm fails with an read-error */
         if( retcode == SCIP_READERROR )
         {
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, exprtree) );
         }
         SCIP_CALL( retcode );

         /* append newly created term */
         SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, *exprtree, termtree, coef) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &termtree) );

         /* find next symbol */
         expr = *newpos;
         while( isspace((unsigned char)*expr) )
            ++expr;
      } while( *expr == '+' || *expr == '-' );
   }
   else
   {
      /* Expr consists of this unique ['+' | '-'] Term */
      if( sign  < 0.0 )
      {
         assert(sign == -1.0);
         SCIP_CALL( SCIPcreateConsExprExprSum(scip, conshdlr, exprtree, 1, &termtree, &sign, 0.0) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &termtree) );
      }
      else
         *exprtree = termtree;
   }

   *newpos = expr;

   return SCIP_OKAY;
}

/** @} */  /* end of parsing methods */

/** given a cons_expr expression, creates an equivalent classic (nlpi-) expression */
static
SCIP_RETCODE makeClassicExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*   sourceexpr,         /**< expression to convert */
   SCIP_EXPR**           targetexpr,         /**< buffer to store pointer to created expression */
   SCIP_CONSEXPR_EXPR**  varexprs,           /**< variable expressions that might occur in expr, their position in this array determines the varidx */
   int                   nvarexprs           /**< number of variable expressions */
   )
{
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr;
   SCIP_EXPR** children = NULL;
   int nchildren;
   int c;

   assert(scip != NULL);
   assert(sourceexpr != NULL);
   assert(targetexpr != NULL);

   exprhdlr = SCIPgetConsExprExprHdlr(sourceexpr);
   nchildren = SCIPgetConsExprExprNChildren(sourceexpr);

   /* collect children expressions from children, if any */
   if( nchildren > 0 )
   {
      SCIP_CALL( SCIPallocBufferArray(scip, &children, nchildren) );
      for( c = 0; c < nchildren; ++c )
      {
         SCIP_CALL( makeClassicExpr(scip, SCIPgetConsExprExprChildren(sourceexpr)[c], &children[c], varexprs, nvarexprs) );
         assert(children[c] != NULL);
      }
   }

   /* create target expression */
   if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr), "var") == 0 )
   {
      int varidx;

      /* find variable expression in varexprs array
       * the position in the array determines the index of the variable in the classic expression
       * TODO if varexprs are sorted, then can do this more efficient
       */
      for( varidx = 0; varidx < nvarexprs; ++varidx )
         if( varexprs[varidx] == sourceexpr )
            break;
      assert(varidx < nvarexprs);

      SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), targetexpr, SCIP_EXPR_VARIDX, varidx) );
   }
   else if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr), "val") == 0 )
   {
      SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), targetexpr, SCIP_EXPR_CONST, SCIPgetConsExprExprValueValue(sourceexpr)) );
   }
   else if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr), "sum") == 0 )
   {
      SCIP_CALL( SCIPexprCreateLinear(SCIPblkmem(scip), targetexpr, nchildren, children, SCIPgetConsExprExprSumCoefs(sourceexpr), SCIPgetConsExprExprSumConstant(sourceexpr)) );
   }
   else if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr), "pow") == 0 )
   {
      assert(nchildren == 1);
      assert(children != NULL && children[0] != NULL);
      SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), targetexpr, SCIP_EXPR_REALPOWER, *children,
            SCIPgetConsExprExprPowExponent(sourceexpr)) );
   }
   else if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr), "prod") == 0 )
   {
      SCIP_EXPRDATA_MONOMIAL* monomial;
      SCIP_CALL( SCIPexprCreateMonomial(SCIPblkmem(scip), &monomial, SCIPgetConsExprExprProductCoef(sourceexpr), nchildren, NULL, NULL) );
      SCIP_CALL( SCIPexprCreatePolynomial(SCIPblkmem(scip), targetexpr, nchildren, children, 1, &monomial, 0.0, FALSE) );
   }
   else if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr), "abs") == 0 )
   {
      assert(nchildren == 1);
      assert(children != NULL && children[0] != NULL);
      SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), targetexpr, SCIP_EXPR_ABS, children[0]) );
   }
   else if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr), "exp") == 0 )
   {
      assert(nchildren == 1);
      assert(children != NULL && children[0] != NULL);
      SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), targetexpr, SCIP_EXPR_EXP, children[0]) );
   }
   else if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr), "log") == 0 )
   {
      assert(nchildren == 1);
      assert(children != NULL && children[0] != NULL);
      SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), targetexpr, SCIP_EXPR_LOG, children[0]) );
   }
   else if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr), "sin") == 0 )
   {
      assert(nchildren == 1);
      assert(children != NULL && children[0] != NULL);
      SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), targetexpr, SCIP_EXPR_SIN, children[0]) );
   }
   else if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr), "cos") == 0 )
   {
      assert(nchildren == 1);
      assert(children != NULL && children[0] != NULL);
      SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), targetexpr, SCIP_EXPR_COS, children[0]) );
   }
   else if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr), "entropy") == 0 )
   {
      SCIP_EXPR* childcopy;
      SCIP_Real minusone = -1.0;

      assert(nchildren == 1);
      assert(children != NULL && children[0] != NULL);

      SCIP_CALL( SCIPexprCopyDeep(SCIPblkmem(scip), &childcopy, children[0]) );
      SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &childcopy, SCIP_EXPR_LOG, childcopy) );
      SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), targetexpr, SCIP_EXPR_MUL, children[0], childcopy) );
      SCIP_CALL( SCIPexprCreateLinear(SCIPblkmem(scip), targetexpr, 1, targetexpr, &minusone, 0.0) );
   }
   else
   {
      SCIPerrorMessage("unsupported expression handler <%s>, cannot convert to classical expression\n", SCIPgetConsExprExprHdlrName(exprhdlr));
      return SCIP_ERROR;
   }

   SCIPfreeBufferArrayNull(scip, &children);

   return SCIP_OKAY;
}

/** create a nonlinear row representation of an expr constraint and stores them in consdata */
static
SCIP_RETCODE createNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< expression constraint */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP_EXPRTREE* exprtree;
   SCIP_EXPR* classicexpr = NULL;
   SCIP_VAR** nlvars = NULL;
   int nnlvars = 0;
   int i;

   assert(scip != NULL);
   assert(cons != NULL);

   conshdlrdata = SCIPconshdlrGetData(SCIPconsGetHdlr(cons));
   assert(conshdlrdata != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   if( consdata->nlrow != NULL )
   {
      SCIP_CALL( SCIPreleaseNlRow(scip, &consdata->nlrow) );
   }

   /* @todo pass correct curvature */
   SCIP_CALL( SCIPcreateNlRow(scip, &consdata->nlrow, SCIPconsGetName(cons), 0.0,
         0, NULL, NULL, 0, NULL, 0, NULL, NULL, consdata->lhs, consdata->rhs, SCIP_EXPRCURV_UNKNOWN) );

   if( consdata->expr == NULL )
      return SCIP_OKAY;

   if( SCIPgetConsExprExprHdlr(consdata->expr) == conshdlrdata->exprsumhdlr )
   {
      /* if root is a sum, then split into linear, quadratic, and expression */
      SCIP_CONSEXPR_EXPR* child;
      SCIP_Real* coefs;

      /* constant term of sum */
      SCIP_CALL( SCIPchgNlRowConstant(scip, consdata->nlrow, SCIPgetConsExprExprSumConstant(consdata->expr)) );

      coefs = SCIPgetConsExprExprSumCoefs(consdata->expr);

      for( i = 0; i < SCIPgetConsExprExprNChildren(consdata->expr); ++i )
      {
         child = SCIPgetConsExprExprChildren(consdata->expr)[i];

         if( SCIPisConsExprExprVar(child) )
         {
            /* linear term */
            SCIP_CALL( SCIPaddLinearCoefToNlRow(scip, consdata->nlrow, SCIPgetConsExprExprVarVar(child), coefs[i]) );
         }
         else if( SCIPgetConsExprExprHdlr(child) == conshdlrdata->exprpowhdlr &&
            SCIPgetConsExprExprPowExponent(child) == 2.0 &&
            SCIPisConsExprExprVar(SCIPgetConsExprExprChildren(child)[0]) )
         {
            /* square term  */
            SCIP_QUADELEM quadelem;

            quadelem.idx1 = SCIPnlrowSearchQuadVar(consdata->nlrow, SCIPgetConsExprExprVarVar(SCIPgetConsExprExprChildren(child)[0]));
            if( quadelem.idx1 == -1 )
            {
               SCIP_CALL( SCIPaddQuadVarToNlRow(scip, consdata->nlrow, SCIPgetConsExprExprVarVar(SCIPgetConsExprExprChildren(child)[0])) );
               quadelem.idx1 = SCIPnlrowGetNQuadVars(consdata->nlrow)-1;
            }
            quadelem.idx2 = quadelem.idx1;
            quadelem.coef = coefs[i];

            SCIP_CALL( SCIPaddQuadElementToNlRow(scip, consdata->nlrow, quadelem) );
         }
         else if( SCIPgetConsExprExprHdlr(child) == conshdlrdata->exprprodhdlr &&
            SCIPgetConsExprExprNChildren(child) == 2 &&
            SCIPisConsExprExprVar(SCIPgetConsExprExprChildren(child)[0]) &&
            SCIPisConsExprExprVar(SCIPgetConsExprExprChildren(child)[1]) )
         {
            /* bilinear term */
            SCIP_QUADELEM quadelem;

            quadelem.idx1 = SCIPnlrowSearchQuadVar(consdata->nlrow, SCIPgetConsExprExprVarVar(SCIPgetConsExprExprChildren(child)[0]));
            if( quadelem.idx1 == -1 )
            {
               SCIP_CALL( SCIPaddQuadVarToNlRow(scip, consdata->nlrow, SCIPgetConsExprExprVarVar(SCIPgetConsExprExprChildren(child)[0])) );
               quadelem.idx1 = SCIPnlrowGetNQuadVars(consdata->nlrow)-1;
            }

            quadelem.idx2 = SCIPnlrowSearchQuadVar(consdata->nlrow, SCIPgetConsExprExprVarVar(SCIPgetConsExprExprChildren(child)[1]));
            if( quadelem.idx2 == -1 )
            {
               SCIP_CALL( SCIPaddQuadVarToNlRow(scip, consdata->nlrow, SCIPgetConsExprExprVarVar(SCIPgetConsExprExprChildren(child)[1])) );
               quadelem.idx2 = SCIPnlrowGetNQuadVars(consdata->nlrow)-1;
            }

            quadelem.coef = coefs[i];

            SCIP_CALL( SCIPaddQuadElementToNlRow(scip, consdata->nlrow, quadelem) );
         }
         else
         {
            /* general nonlinear term */
            SCIP_EXPR* classicchild;

            /* make classic expression of child i */
            SCIP_CALL( makeClassicExpr(scip, child, &classicchild, consdata->varexprs, consdata->nvarexprs) );

            /* create or extend classicexpr */
            if( classicexpr == NULL )
            {
               SCIP_CALL( SCIPexprCreateLinear(SCIPblkmem(scip), &classicexpr, 1, &classicchild, coefs + i, 0.0) );
            }
            else
            {
               SCIP_CALL( SCIPexprAddToLinear(SCIPblkmem(scip), classicexpr, 1, &coefs[i], &classicchild, 0.0) );
            }
         }
      }

      if( classicexpr != NULL )
      {
         /* reindex variables in classicexpr so that only used variables are left */
         int* varsusage;
         int* reindexvars;

         /* get count how often variables are used in expr */
         SCIP_CALL( SCIPallocBufferArray(scip, &varsusage, consdata->nvarexprs) );
         BMSclearMemoryArray(varsusage, consdata->nvarexprs);
         SCIPexprGetVarsUsage(classicexpr, varsusage);

         /* sort out unused variables and collect and reindex remaining variables */
         SCIP_CALL( SCIPallocBufferArray(scip, &reindexvars, consdata->nvarexprs) );
         SCIP_CALL( SCIPallocBufferArray(scip, &nlvars, consdata->nvarexprs) );
         nnlvars = 0;
         for( i = 0; i < consdata->nvarexprs; ++i )
         {
            if( varsusage[i] == 0 )
            {
               reindexvars[i] = -1;
            }
            else
            {
               reindexvars[i] = nnlvars;
               nlvars[nnlvars] = SCIPgetConsExprExprVarVar(consdata->varexprs[i]);
               ++nnlvars;
            }
         }

         SCIPexprReindexVars(classicexpr, reindexvars);

         SCIPfreeBufferArray(scip, &varsusage);
         SCIPfreeBufferArray(scip, &reindexvars);
      }
   }
   else if( SCIPgetConsExprExprHdlr(consdata->expr) == conshdlrdata->exprpowhdlr &&
      SCIPgetConsExprExprPowExponent(consdata->expr) == 2.0 &&
      SCIPisConsExprExprVar(SCIPgetConsExprExprChildren(consdata->expr)[0]) )
   {
      /* if root is a x^2, then set the quadratic part of the nlrow */
      SCIP_QUADELEM quadelem;

      SCIP_CALL( SCIPaddQuadVarToNlRow(scip, consdata->nlrow, SCIPgetConsExprExprVarVar(SCIPgetConsExprExprChildren(consdata->expr)[0])) );
      quadelem.idx1 = 0;
      quadelem.idx2 = 0;
      quadelem.coef = 1.0;

      SCIP_CALL( SCIPaddQuadElementToNlRow(scip, consdata->nlrow, quadelem) );
   }
   else if( SCIPgetConsExprExprHdlr(consdata->expr) == conshdlrdata->exprprodhdlr &&
      SCIPgetConsExprExprNChildren(consdata->expr) == 2 &&
      SCIPisConsExprExprVar(SCIPgetConsExprExprChildren(consdata->expr)[0]) &&
      SCIPisConsExprExprVar(SCIPgetConsExprExprChildren(consdata->expr)[1]) )
   {
      /* if root is a bilinear term x*y, then set the quadratic part of the nlrow */
      SCIP_QUADELEM quadelem;

      SCIP_CALL( SCIPaddQuadVarToNlRow(scip, consdata->nlrow, SCIPgetConsExprExprVarVar(SCIPgetConsExprExprChildren(consdata->expr)[0])) );
      SCIP_CALL( SCIPaddQuadVarToNlRow(scip, consdata->nlrow, SCIPgetConsExprExprVarVar(SCIPgetConsExprExprChildren(consdata->expr)[1])) );

      quadelem.idx1 = 0;
      quadelem.idx2 = 1;
      quadelem.coef = 1.0;

      SCIP_CALL( SCIPaddQuadElementToNlRow(scip, consdata->nlrow, quadelem) );
   }
   else
   {
      /* make classic expression */
      SCIP_CALL( makeClassicExpr(scip, consdata->expr, &classicexpr, consdata->varexprs, consdata->nvarexprs) );

      /* collect variables */
      SCIP_CALL( SCIPallocBufferArray(scip, &nlvars, consdata->nvarexprs) );

      nnlvars = consdata->nvarexprs;
      for( i = 0; i < consdata->nvarexprs; ++i )
         nlvars[i] = SCIPgetConsExprExprVarVar(consdata->varexprs[i]);
   }
   assert((classicexpr != NULL) == (nlvars != NULL));

   if( classicexpr != NULL )
   {
      /* make classic expression tree */
      SCIP_CALL( SCIPexprtreeCreate(SCIPblkmem(scip), &exprtree, classicexpr, nnlvars, 0, NULL) );

      /* set variables in expression tree */
      SCIP_CALL( SCIPexprtreeSetVars(exprtree, nnlvars, nlvars) );
      SCIPfreeBufferArray(scip, &nlvars);

      /* add expression tree in nlrow (this will make a copy) */
      SCIP_CALL( SCIPsetNlRowExprtree(scip, consdata->nlrow, exprtree) );

      /* free exprtree */
      SCIP_CALL( SCIPexprtreeFree(&exprtree) );
   }

   return SCIP_OKAY;
}

/** computes the branching scores for a given set of constraints; the scores are computed by computing the violation of
 *  each expression by considering the values of the linearization variables of the expression and its children
 *
 *  @note function assumes that violations have been computed
 */
static
SCIP_RETCODE computeBranchingScores(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< nonlinear constraints handler */
   SCIP_CONS**           conss,              /**< constraints */
   int                   nconss,             /**< number of constraints */
   SCIP_Real             minviolation,       /**< minimal violation in expression to register a branching score */
   SCIP_Bool             evalauxvalues,      /**< whether auxiliary values of expressions need to be evaluated */
   SCIP_SOL*             sol,                /**< solution to branch on (NULL for LP solution) */
   unsigned int          soltag              /**< solution tag */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSEXPR_EXPR* expr;
   SCIP_Real auxvarvalue;
   SCIP_Bool overestimate;
   SCIP_Bool underestimate;
   unsigned int brscoretag;
   int e;
   int i;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(conss != NULL || nconss == 0);
   assert(nconss >= 0);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* tag to tell whether branching scores in expression belong to this sweep */
   brscoretag = ++(conshdlrdata->lastbrscoretag);

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, NULL, SCIP_CONSEXPRITERATOR_DFS, FALSE) );

   /* call branching score callbacks for expressions in violated constraints */
   for( i = 0; i < nconss; ++i )
   {
      assert(conss != NULL);
      assert(conss[i] != NULL);

      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      /* skip satisfied constraints */
      if( !SCIPisGT(scip, consdata->lhsviol, SCIPfeastol(scip)) && !SCIPisGT(scip, consdata->rhsviol, SCIPfeastol(scip)) )
         continue;

      consdata->expr->brscore = 0.0;  /* TODO why do we need this? */

      for( expr = SCIPexpriteratorRestartDFS(it, consdata->expr); !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
      {
         /* if no auxvar, then no need to compute branching score here (nothing can be violated) */
         if( expr->auxvar == NULL )
            continue;

         /* make sure expression has been evaluated, so evalvalue makes sense */
         SCIP_CALL( SCIPevalConsExprExpr(scip, conshdlr, expr, sol, soltag) );

         auxvarvalue = SCIPgetSolVal(scip, sol, expr->auxvar);

         /* compute violation w.r.t. original variables */
         if( expr->evalvalue != SCIP_INVALID ) /*lint !e777*/
         {
            /* the expression could be evaluated, then look on which side it is violated */

            /* first, violation of auxvar <= expr, which is violated if auxvar - expr > 0 */
            overestimate = SCIPgetConsExprExprNLocksNeg(expr) > 0 && auxvarvalue - expr->evalvalue > minviolation;

            /* next, violation of auxvar >= expr, which is violated if expr - auxvar > 0 */
            underestimate = SCIPgetConsExprExprNLocksPos(expr) > 0 && expr->evalvalue - auxvarvalue > minviolation;
         }
         else
         {
            /* if expression could not be evaluated, then both under- and overestimate should be considered */
            overestimate = SCIPgetConsExprExprNLocksNeg(expr) > 0;
            underestimate = SCIPgetConsExprExprNLocksPos(expr) > 0;
         }

         if( !overestimate && !underestimate )
            continue;

         /* if there is violation, then consider branching
          * call branching score callbacks of all nlhdlrs
          */
         for( e = 0; e < expr->nenfos; ++e )
         {
            SCIP_CONSEXPR_NLHDLR* nlhdlr;
            SCIP_Bool nlhdlrsuccess;

            nlhdlr = expr->enfos[e]->nlhdlr;
            assert(nlhdlr != NULL);

            /* update auxvalue as corresponding to nlhdlr, if necessary */
            if( evalauxvalues )
            {
               SCIP_CALL( SCIPevalauxConsExprNlhdlr(scip, nlhdlr, expr, expr->enfos[e]->nlhdlrexprdata, &expr->enfos[e]->auxvalue, sol) );
            }

            /* if there is violation w.r.t. auxiliary variables, then call brscore of nlhdlr
             * the nlhdlr currently needs to recheck whether auxvar <= expr or auxvar >= expr is violated
             * and whether that corresponds to the relation that the nlhdlr tries to enforce
             */
            if( expr->enfos[e]->auxvalue == SCIP_INVALID ||  /*lint !e777*/
               (overestimate && auxvarvalue - expr->enfos[e]->auxvalue > minviolation) ||
               (underestimate && expr->enfos[e]->auxvalue - auxvarvalue > minviolation) )
            {
               SCIP_CALL( SCIPbranchscoreConsExprNlHdlr(scip, nlhdlr, expr, expr->enfos[e]->nlhdlrexprdata, sol, expr->enfos[e]->auxvalue, brscoretag, &nlhdlrsuccess) );
               SCIPdebugMsg(scip, "branchscore of nlhdlr %s for expr %p (%s) with auxviolation %g: success = %d\n", nlhdlr->name, expr, expr->exprhdlr->name, REALABS(expr->enfos[e]->auxvalue - auxvarvalue), nlhdlrsuccess);
               /* if( nlhdlrsuccess )
                     success = TRUE; */
            }
         }
         /* if noone had success, then the violation here is caused by a violation deeper down in the expression tree,
          * so there was no need to add branching scores from this expression
          */
      }
   }

   /* propagate branching score callbacks from expressions with children to variable expressions */
   for( i = 0; i < nconss; ++i )
   {
      assert(conss != NULL);
      assert(conss[i] != NULL);

      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      /* for satisfied constraints, no branching score has been computed, so no need to propagte from here */
      if( !SCIPisGT(scip, consdata->lhsviol, SCIPfeastol(scip)) && !SCIPisGT(scip, consdata->rhsviol, SCIPfeastol(scip)) )
         continue;

      /* we need to allow revisiting here, as we always want to propagate branching scores to the variable expressions */
      SCIP_CALL( SCIPexpriteratorInit(it, consdata->expr, SCIP_CONSEXPRITERATOR_DFS, TRUE) );
      SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_VISITINGCHILD | SCIP_CONSEXPRITERATOR_LEAVEEXPR);

      for( expr = SCIPexpriteratorGetCurrent(it); !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
      {
         switch( SCIPexpriteratorGetStageDFS(it) )
         {
            case SCIP_CONSEXPRITERATOR_VISITINGCHILD :
            {
               /* propagate branching score, if any, from this expression to current child
                * NOTE: this only propagates down branching scores that were computed by computeBranchScore
                * we use the brscoretag to recognize whether this expression has a valid branching score
                */
               if( expr->brscoretag == brscoretag )
                  SCIPaddConsExprExprBranchScore(scip, SCIPexpriteratorGetChildExprDFS(it), brscoretag, expr->brscore);

               break;
            }

            case SCIP_CONSEXPRITERATOR_LEAVEEXPR :
            {
               /* invalidate the branching scores in this expression, so they are not passed on in case this expression
                * is visited again
                * do this only for expressions with children, since for variables we need the brscoretag to be intact
                */
               if( expr->nchildren > 0 )
                  expr->brscoretag = 0;

               break;
            }

            default:
               SCIPABORT();
               break;
         }
      }
   }

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** registers branching candidates */
static
SCIP_RETCODE registerBranchingCandidates(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< nonlinear constraints handler */
   SCIP_CONS**           conss,              /**< constraints to check */
   int                   nconss,             /**< number of constraints to check */
   SCIP_SOL*             sol,                /**< solution to branch on (NULL for LP solution) */
   unsigned int          soltag,             /**< solution tag */
   SCIP_Real             minviolation,       /**< minimal violation in expression to register a branching score */
   SCIP_Bool             evalauxvalues,      /**< whether auxiliary values of expressions need to be evaluated */
   int*                  nnotify             /**< counter for number of notifications performed */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP_VAR* var;
   int c;
   int i;

   assert(conshdlr != NULL);
   assert(conss != NULL || nconss == 0);
   assert(nnotify != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   *nnotify = 0;

   /* compute branching scores by considering violation of all expressions */
   SCIP_CALL( computeBranchingScores(scip, conshdlr, conss, nconss, minviolation, evalauxvalues, sol, soltag) );

   for( c = 0; c < nconss; ++c )
   {
      assert(conss != NULL && conss[c] != NULL);

      consdata = SCIPconsGetData(conss[c]);
      assert(consdata != NULL);

      /* consider only violated constraints */
      if( SCIPisGT(scip, consdata->lhsviol, SCIPfeastol(scip)) || SCIPisGT(scip, consdata->rhsviol, SCIPfeastol(scip)) )
      {
         assert(consdata->varexprs != NULL);

         for( i = 0; i < consdata->nvarexprs; ++i )
         {
            SCIP_Real brscore;

            /* skip variable expressions that do not have a valid branching score (contained in no currently violated constraint) */
            if( conshdlrdata->lastbrscoretag != consdata->varexprs[i]->brscoretag )
               continue;

            brscore = consdata->varexprs[i]->brscore;
            var = SCIPgetConsExprExprVarVar(consdata->varexprs[i]);
            assert(var != NULL);

            /* introduce variable if it has not been fixed yet and has a branching score > 0 */
            if( !SCIPisEQ(scip, SCIPcomputeVarLbLocal(scip, var), SCIPcomputeVarUbLocal(scip, var)) )
            {
               SCIPdebugMsg(scip, "add variable <%s>[%g,%g] as extern branching candidate with score %g\n", SCIPvarGetName(var), SCIPcomputeVarLbLocal(scip, var), SCIPcomputeVarUbLocal(scip, var), brscore);

               SCIP_CALL( SCIPaddExternBranchCand(scip, var, brscore, SCIP_INVALID) );
               ++(*nnotify);
            }
         }
      }
   }

   return SCIP_OKAY;
}

/** registers all unfixed variables in violated constraints as branching candidates */
static
SCIP_RETCODE registerBranchingCandidatesAllUnfixed(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< nonlinear constraints handler */
   SCIP_CONS**           conss,              /**< constraints */
   int                   nconss,             /**< number of constraints */
   int*                  nnotify             /**< counter for number of notifications performed */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_VAR* var;
   int c;
   int i;

   assert(conshdlr != NULL);
   assert(conss != NULL || nconss == 0);
   assert(nnotify != NULL);

   *nnotify = 0;

   for( c = 0; c < nconss; ++c )
   {
      assert(conss != NULL && conss[c] != NULL);

      consdata = SCIPconsGetData(conss[c]);
      assert(consdata != NULL);

      /* consider only violated constraints */
      if( !SCIPisGT(scip, consdata->lhsviol, SCIPfeastol(scip)) && !SCIPisGT(scip, consdata->rhsviol, SCIPfeastol(scip)) )
         continue;

      /* register all variables that have not been fixed yet */
      assert(consdata->varexprs != NULL);
      for( i = 0; i < consdata->nvarexprs; ++i )
      {
         var = SCIPgetConsExprExprVarVar(consdata->varexprs[i]);
         assert(var != NULL);

         if( !SCIPisEQ(scip, SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var)) )
         {
            SCIP_CALL( SCIPaddExternBranchCand(scip, var, MAX(consdata->lhsviol, consdata->rhsviol), SCIP_INVALID) );
            ++(*nnotify);
         }
      }
   }

   return SCIP_OKAY;
}

/** compares nonlinear handler by priority
 *
 * if handlers have same priority, then compare by name
 */
static
int nlhdlrCmp(
   void*                 hdlr1,              /**< first handler */
   void*                 hdlr2               /**< second handler */
   )
{
   SCIP_CONSEXPR_NLHDLR* h1;
   SCIP_CONSEXPR_NLHDLR* h2;

   assert(hdlr1 != NULL);
   assert(hdlr2 != NULL);

   h1 = (SCIP_CONSEXPR_NLHDLR*)hdlr1;
   h2 = (SCIP_CONSEXPR_NLHDLR*)hdlr2;

   if( h1->priority != h2->priority )
      return (int)(h1->priority - h2->priority);

   return strcmp(h1->name, h2->name);
}

/** frees auxiliary variables which have been added to compute an outer approximation */
static
SCIP_RETCODE freeAuxVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONS**           conss,              /**< constraints to check for auxiliary variables */
   int                   nconss              /**< total number of constraints */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSEXPR_EXPR* expr;
   SCIP_CONSDATA* consdata;
   int i;

   assert(conss != NULL || nconss == 0);
   assert(nconss >= 0);

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, NULL, SCIP_CONSEXPRITERATOR_DFS, FALSE) );

   for( i = 0; i < nconss; ++i )
   {
      assert(conss != NULL);
      assert(conss[i] != NULL);

      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      if( consdata->expr == NULL )
         continue;

      for( expr = SCIPexpriteratorRestartDFS(it, consdata->expr); !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
      {
         SCIP_CALL( freeAuxVar(scip, expr) );
      }
   }

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** calls separation initialization callback for each expression */
static
SCIP_RETCODE initSepa(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< nonlinear constraints handler */
   SCIP_CONS**           conss,              /**< constraints */
   int                   nconss,             /**< number of constraints */
   SCIP_Bool*            infeasible          /**< pointer to store whether the problem is infeasible or not */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSEXPR_EXPR* expr;
   int c, e;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(conss != NULL || nconss == 0);
   assert(nconss >= 0);
   assert(infeasible != NULL);

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, NULL, SCIP_CONSEXPRITERATOR_DFS, FALSE) );

   *infeasible = FALSE;
   for( c = 0; c < nconss && !*infeasible; ++c )
   {
      assert(conss != NULL);
      assert(conss[c] != NULL);

      /* call separation initialization callback for 'initial' constraints only */
      if( !SCIPconsIsInitial(conss[c]) )
         continue;

      consdata = SCIPconsGetData(conss[c]);
      assert(consdata != NULL);
      assert(consdata->expr != NULL);

      for( expr = SCIPexpriteratorRestartDFS(it, consdata->expr); !SCIPexpriteratorIsEnd(it) && !*infeasible; expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
      {
         /* call initsepa of all nlhdlrs in expr */
         for( e = 0; e < expr->nenfos; ++e )
         {
            SCIP_CONSEXPR_NLHDLR* nlhdlr;
            SCIP_Bool underestimate;
            SCIP_Bool overestimate;
            assert(expr->enfos[e] != NULL);

            nlhdlr = expr->enfos[e]->nlhdlr;
            assert(nlhdlr != NULL);

            /* only init sepa if there is an initsepa callback */
            if( !SCIPhasConsExprNlhdlrInitSepa(nlhdlr) )
               continue;

            assert(!expr->enfos[e]->issepainit);

            /* check whether expression needs to be under- or overestimated */
            overestimate = SCIPgetConsExprExprNLocksNeg(expr) > 0;
            underestimate = SCIPgetConsExprExprNLocksPos(expr) > 0;
            assert(underestimate || overestimate);

            /* call the separation initialization callback of the nonlinear handler */
            SCIP_CALL( SCIPinitsepaConsExprNlhdlr(scip, conshdlr, nlhdlr, expr, expr->enfos[e]->nlhdlrexprdata,
               overestimate, underestimate, infeasible) );
            expr->enfos[e]->issepainit = TRUE;

            if( *infeasible )
            {
               /* stop everything if we detected infeasibility */
               SCIPdebugMsg(scip, "detect infeasibility for constraint %s during initsepa()\n", SCIPconsGetName(conss[c]));
               break;
            }
         }
      }
   }

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** call separation or estimator callback of nonlinear handler
 *
 * Calls the separation callback, if available.
 * Otherwise, calls the estimator callback, if available, and constructs a cut from the estimator.
 */
static
SCIP_RETCODE separatePointExprNlhdlr(
   SCIP*                 scip,               /**< SCIP main data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONSEXPR_NLHDLR* nlhdlr,             /**< nonlinear handler */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression */
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata, /**< nonlinear handler data of expression */
   SCIP_SOL*             sol,                /**< solution to be separated (NULL for the LP solution) */
   SCIP_Real             auxvalue,           /**< current value of expression w.r.t. auxiliary variables as obtained from EVALAUX */
   SCIP_Bool             overestimate,       /**< whether the expression needs to be over- or underestimated */
   SCIP_Real             mincutviolation,    /**< minimal violation of a cut if it should be added to the LP */
   SCIP_Bool             separated,          /**< whether another nonlinear handler already added a cut for this expression */
   SCIP_RESULT*          result              /**< pointer to store the result */
   )
{
   int ncuts;

   assert(result != NULL);

   /* call separation callback of the nlhdlr */
   SCIP_CALL( SCIPsepaConsExprNlhdlr(scip, conshdlr, nlhdlr, expr, nlhdlrexprdata, sol, auxvalue, overestimate, mincutviolation, separated, result, &ncuts) );

   /* if it was not running (e.g., because it was not available) or did not find anything, then try with estimator callback */
   if( *result != SCIP_DIDNOTRUN && *result != SCIP_DIDNOTFIND )
      return SCIP_OKAY;

   /* now call the estimator callback of the nlhdlr */
   if( SCIPhasConsExprNlhdlrEstimate(nlhdlr) )
   {
      SCIP_ROWPREP* rowprep;
      SCIP_VAR* auxvar;
      SCIP_Bool success = FALSE;

      *result = SCIP_DIDNOTFIND;

      SCIP_CALL( SCIPcreateRowprep(scip, &rowprep, overestimate ? SCIP_SIDETYPE_LEFT : SCIP_SIDETYPE_RIGHT, TRUE) );

      auxvar = SCIPgetConsExprExprAuxVar(expr);
      assert(auxvar != NULL);

      SCIP_CALL( SCIPestimateConsExprNlhdlr(scip, conshdlr, nlhdlr, expr, nlhdlrexprdata, sol, auxvalue, overestimate, SCIPgetSolVal(scip, sol, auxvar), rowprep, &success) );

      /* complete estimator to cut and clean it up */
      if( success )
      {
         SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, auxvar, -1.0) );

         SCIP_CALL( SCIPcleanupRowprep(scip, rowprep, sol, SCIP_CONSEXPR_CUTMAXRANGE, mincutviolation, NULL, &success) );
      }

      /* if cut looks good (numerics ok and cutting off solution), then turn into row and add to sepastore */
      if( success )
      {
         SCIP_ROW* row;
         SCIP_Bool infeasible;

         SCIP_CALL( SCIPgetRowprepRowCons(scip, &row, rowprep, conshdlr) );

#ifdef SCIP_DEBUG
         SCIPdebugMsg(scip, "adding cut ");
         SCIP_CALL( SCIPprintRow(scip, row, NULL) );
#endif

         SCIP_CALL( SCIPaddRow(scip, row, FALSE, &infeasible) );

         if( infeasible )
         {
            *result = SCIP_CUTOFF;
            ++nlhdlr->ncutoffs;
         }
         else
         {
            *result = SCIP_SEPARATED;
            ++nlhdlr->ncutsfound;
         }

         SCIP_CALL( SCIPreleaseRow(scip, &row) );
      }

      SCIPfreeRowprep(scip, &rowprep);
   }

   return SCIP_OKAY;
}

/** tries to separate solution or LP solution by a linear cut for a single expression */
static
SCIP_RETCODE separatePointExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraints handler */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression */
   SCIP_SOL*             sol,                /**< solution to separate, or NULL if LP solution should be used */
   unsigned int          soltag,             /**< tag of solution */
   SCIP_Real             minviolation,       /**< minimal violation in an expression to call separation */
   SCIP_Real             mincutviolation,    /**< minimal violation of a cut if it should be added to the LP */
   SCIP_Bool*            separated,          /**< buffer to store whether solution could be separated */
   SCIP_Bool*            infeasible,         /**< buffer to store whether infeasibility to was detected */
   SCIP_Real*            maxauxviolation     /**< buffer to store maximal violation w.r.t. auxiliary variables (in exprs that are violated > minviolation), or NULL if not of interest */
   )
{
   SCIP_Real auxvarvalue;
   SCIP_Bool underestimate;
   SCIP_Bool overestimate;
   SCIP_RESULT hdlrresult;
   int e;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(expr->auxvar != NULL);  /* there must be a variable attached to the expression in order to construct a cut here */
   assert(separated != NULL);
   assert(infeasible != NULL);

   *separated = FALSE;
   *infeasible = FALSE;

   auxvarvalue = SCIPgetSolVal(scip, sol, expr->auxvar);

   /* make sure that this expression has been evaluated */
   SCIP_CALL( SCIPevalConsExprExpr(scip, conshdlr, expr, sol, soltag) );

   /* compute violation and decide whether under- or overestimate is required */
   if( expr->evalvalue != SCIP_INVALID ) /*lint !e777*/
   {
      /* the expression could be evaluated, then look how much and on which side it is violated */

      /* first, violation of auxvar <= expr, which is violated if auxvar - expr > 0 */
      overestimate = SCIPgetConsExprExprNLocksNeg(expr) > 0 && auxvarvalue - expr->evalvalue > minviolation;

      /* next, violation of auxvar >= expr, which is violated if expr - auxvar > 0 */
      underestimate = SCIPgetConsExprExprNLocksPos(expr) > 0 && expr->evalvalue - auxvarvalue > minviolation;
   }
   else
   {
      /* if expression could not be evaluated, then both under- and overestimate should be considered */
      overestimate = SCIPgetConsExprExprNLocksNeg(expr) > 0;
      underestimate = SCIPgetConsExprExprNLocksPos(expr) > 0;
   }

   /* no sufficient violation w.r.t. the original variables -> skip expression */
   if( !overestimate && !underestimate )
      return SCIP_OKAY;

   /* call the separation callbacks of the nonlinear handlers */
   for( e = 0; e < expr->nenfos; ++e )
   {
      SCIP_CONSEXPR_NLHDLR* nlhdlr;

      nlhdlr = expr->enfos[e]->nlhdlr;
      assert(nlhdlr != NULL);

      /* evaluate the expression w.r.t. the nlhdlrs auxiliary variables */
      SCIP_CALL( SCIPevalauxConsExprNlhdlr(scip, nlhdlr, expr, expr->enfos[e]->nlhdlrexprdata, &expr->enfos[e]->auxvalue, sol) );

      if( maxauxviolation != NULL )
      {
         /* update maxauxviolation */
         if( expr->enfos[e]->auxvalue == SCIP_INVALID )  /*lint !e777*/
            *maxauxviolation = SCIPinfinity(scip);
         else if( overestimate && auxvarvalue - expr->enfos[e]->auxvalue > *maxauxviolation )
            *maxauxviolation = auxvarvalue - expr->enfos[e]->auxvalue;
         else if( underestimate && expr->enfos[e]->auxvalue - auxvarvalue > *maxauxviolation )
            *maxauxviolation = expr->enfos[e]->auxvalue - auxvarvalue;
      }

      SCIPdebugMsg(scip, "sepa of nlhdlr <%s> for expr %p (%s) with auxviolation %g origviolation %g under:%d over:%d\n", nlhdlr->name, expr, expr->exprhdlr->name, REALABS(expr->enfos[e]->auxvalue - auxvarvalue), REALABS(expr->evalvalue - auxvarvalue), underestimate, overestimate);

      /* if we want overestimation and violation w.r.t. auxiliary variables is also present, then call separation of nlhdlr */
      if( overestimate && (expr->enfos[e]->auxvalue == SCIP_INVALID || auxvarvalue - expr->enfos[e]->auxvalue > minviolation) )  /*lint !e777*/
      {
         /* call the separation or estimation callback of the nonlinear handler for overestimation */
         hdlrresult = SCIP_DIDNOTFIND;
         SCIP_CALL( separatePointExprNlhdlr(scip, conshdlr, nlhdlr, expr, expr->enfos[e]->nlhdlrexprdata, sol,
            expr->enfos[e]->auxvalue, TRUE, mincutviolation, *separated, &hdlrresult) );

         if( hdlrresult == SCIP_CUTOFF )
         {
            SCIPdebugMsg(scip, "found a cutoff -> stop separation\n");
            *infeasible = TRUE;
            break;
         }

         if( hdlrresult == SCIP_SEPARATED )
         {
            SCIPdebugMsg(scip, "nlhdlr <%s> separating the current solution\n", nlhdlr->name);
            *separated = TRUE;
            /* TODO or should we always just stop here? */
         }
      }

      if( underestimate && (expr->enfos[e]->auxvalue == SCIP_INVALID || expr->enfos[e]->auxvalue - auxvarvalue > minviolation) )  /*lint !e777*/
      {
         /* call the separation or estimation callback of the nonlinear handler for underestimation */
         hdlrresult = SCIP_DIDNOTFIND;
         SCIP_CALL( separatePointExprNlhdlr(scip, conshdlr, nlhdlr, expr, expr->enfos[e]->nlhdlrexprdata, sol,
            expr->enfos[e]->auxvalue, FALSE, mincutviolation, *separated, &hdlrresult) );

         if( hdlrresult == SCIP_CUTOFF )
         {
            SCIPdebugMsg(scip, "found a cutoff -> stop separation\n");
            *infeasible = TRUE;
            break;
         }

         if( hdlrresult == SCIP_SEPARATED )
         {
            SCIPdebugMsg(scip, "nlhdlr <%s> separating the current solution\n", nlhdlr->name);
            *separated = TRUE;
            /* TODO or should we always just stop here? */
         }
      }
   }

   return SCIP_OKAY;
}

/** tries to separate solution or LP solution by a linear cut
 *
 *  assumes that constraint violations have been computed
 */
static
SCIP_RETCODE separatePoint(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< nonlinear constraints handler */
   SCIP_CONS**           conss,              /**< constraints */
   int                   nconss,             /**< number of constraints */
   int                   nusefulconss,       /**< number of constraints that seem to be useful */
   SCIP_SOL*             sol,                /**< solution to separate, or NULL if LP solution should be used */
   unsigned int          soltag,             /**< tag of solution */
   SCIP_Real             minviolation,       /**< minimal violation in an expression to call separation */
   SCIP_Real             mincutviolation,    /**< minimal violation of a cut if it should be added to the LP */
   SCIP_RESULT*          result,             /**< result of separation */
   SCIP_Real*            maxauxviolation     /**< buffer to store maximal violation w.r.t. auxiliary variables (in exprs that are violated > minviolation), or NULL if not of interest */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSEXPR_EXPR* expr;
   SCIP_Bool infeasible;
   SCIP_Bool separated;
   int c;

   assert(conss != NULL || nconss == 0);
   assert(nconss >= nusefulconss);
   assert(mincutviolation >= 0.0);
   assert(result != NULL);

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, NULL, SCIP_CONSEXPRITERATOR_DFS, FALSE) );

   for( c = 0; c < nconss; ++c )
   {
      assert(conss != NULL && conss[c] != NULL);

      consdata = SCIPconsGetData(conss[c]);
      assert(consdata != NULL);

      /* skip constraints that are not enabled */
      if( !SCIPconsIsEnabled(conss[c]) || SCIPconsIsDeleted(conss[c]) )
         continue;
      assert(SCIPconsIsActive(conss[c]));

      /* skip non-violated constraints */
      if( SCIPisLE(scip, MAX(consdata->lhsviol, consdata->rhsviol), SCIPfeastol(scip)) )
         continue;

      #ifdef SEPA_DEBUG
      {
         int i;
         SCIPdebugMsg(scip, "separating point\n");
         for( i = 0; i < consdata->nvarexprs; ++i )
         {
            SCIP_VAR* var;
            var = SCIPgetConsExprExprVarVar(consdata->varexprs[i]);
            SCIPdebugMsg("  %s = %g bounds: %g,%g\n", SCIPvarGetName(var), SCIPgetSolVal(scip, sol, var), SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var));
         }
         SCIPdebugMsg(scip, "in constraint\n");
         SCIP_CALL( SCIPprintCons(scip, conss[c], NULL) );
         SCIPinfoMessage(scip, NULL, ";\n");
      }
      #endif

      for( expr = SCIPexpriteratorRestartDFS(it, consdata->expr); !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
      {
         /* it only makes sense to call the separation callback if there is a variable attached to the expression */
         if( expr->auxvar == NULL )
            continue;

         SCIP_CALL( separatePointExpr(scip, conshdlr, expr, sol, soltag, minviolation, mincutviolation, &separated, &infeasible, maxauxviolation) );

         if( infeasible )
         {
            *result = SCIP_CUTOFF;
            goto TERMINATE;
         }

         if( separated )
            *result = SCIP_SEPARATED;
      }

      /* enforce only useful constraints; others are only checked and enforced if we are still feasible or have not
       * found a separating cut yet
       */
      if( c >= nusefulconss && *result == SCIP_SEPARATED )
         break;
   }

TERMINATE:
   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}


/** helper function to enforce constraints */
static
SCIP_RETCODE enforceConstraints(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONS**           conss,              /**< constraints to process */
   int                   nconss,             /**< number of constraints */
   int                   nusefulconss,       /**< number of useful (non-obsolete) constraints to process */
   SCIP_SOL*             sol,                /**< solution to enforce (NULL for the LP solution) */
   SCIP_RESULT*          result              /**< pointer to store the result of the enforcing call */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP_Real maxviol;
   SCIP_Real minviolation;
   SCIP_Real maxauxviolation = 0.0;
   SCIP_RESULT propresult;
   SCIP_Bool force;
   unsigned int soltag;
   int nnotify;
   int nchgbds;
   int ndelconss;
   int c;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlr != NULL);

   maxviol = 0.0;
   soltag = ++conshdlrdata->lastsoltag;

   /* force tightenings when calling enforcement for the first time for a node */
   force = conshdlrdata->lastenfolpnodenum != SCIPnodeGetNumber(SCIPgetCurrentNode(scip));
   conshdlrdata->lastenfolpnodenum = SCIPnodeGetNumber(SCIPgetCurrentNode(scip));

   for( c = 0; c < nconss; ++c )
   {
      SCIP_CALL( computeViolation(scip, conss[c], NULL, soltag) );
      consdata = SCIPconsGetData(conss[c]);

      /* compute max violation */
      maxviol = MAX3(maxviol, consdata->lhsviol, consdata->rhsviol);
   }
   SCIPdebugMsg(scip, "enforcing constraints with maxviol=%e node %d\n", maxviol, SCIPnodeGetNumber(SCIPgetCurrentNode(scip)));

   *result = SCIPisGT(scip, maxviol, SCIPfeastol(scip)) ? SCIP_INFEASIBLE : SCIP_FEASIBLE;

   if( *result == SCIP_FEASIBLE )
      return SCIP_OKAY;

   /* try to propagate */
   nchgbds = 0;
   ndelconss = 0;
   SCIP_CALL( propConss(scip, conshdlr, conss, nconss, force, &propresult, &nchgbds, &ndelconss) );

   if( propresult == SCIP_CUTOFF || propresult == SCIP_REDUCEDDOM )
   {
      *result = propresult;
      return SCIP_OKAY;
   }

   minviolation = SCIPfeastol(scip);
   do
   {
      SCIPdebugMsg(scip, "enforce by separation for minviolation %g\n", minviolation);

      /* try to separate the LP solution */
      SCIP_CALL( separatePoint(scip, conshdlr, conss, nconss, nusefulconss, sol, soltag, minviolation, SCIPfeastol(scip), result, &maxauxviolation) );

      if( *result == SCIP_CUTOFF || *result == SCIP_SEPARATED )
         return SCIP_OKAY;

      /* find branching candidates */
      SCIP_CALL( registerBranchingCandidates(scip, conshdlr, conss, nconss, sol, soltag, minviolation, FALSE, &nnotify) );
      SCIPdebugMsg(scip, "registered %d external branching candidates\n", nnotify);

      /* if no cut or branching candidate, then try less violated expressions
       * maxauxviolation tells us the maximal value we need to choose to have at least one violation in exprs with current violation > minviolation considered
       * the latter condition means, however, that maxauxviolation = 0 is possible, that is, for all exprs with violation > minviolation, the auxviolation is 0
       * TODO for now, just reduce minviolation by a factor of 10, though taking maxauxviolation into account would be nice
       */
      if( nnotify == 0 )
         minviolation /= 10.0;
   }
   while( nnotify == 0 && minviolation > 1.0/SCIPinfinity(scip) ); /* stopping at SCIPepsilon is not sufficient for numerically difficult instances, but something like 1e-100 doesn't seem useful, too; use 1e-20 for now */

   if( nnotify > 0 )
      return SCIP_OKAY;

   SCIPdebugMsg(scip, "could not enforce violation %g in regular ways, becoming desperate now...\n", maxviol);

   /* could not find branching candidates even when looking at minimal violated (>eps) expressions
    * now look if we find any unfixed variable that we could still branch on
    */
   SCIP_CALL( registerBranchingCandidatesAllUnfixed(scip, conshdlr, conss, nconss, &nnotify) );

   if( nnotify > 0 )
   {
      SCIPdebugMsg(scip, "registered %d unfixed variables as branching candidates", nnotify);
      ++conshdlrdata->ndesperatebranch;

      return SCIP_OKAY;
   }

   /* if everything is fixed in violated constraints, then let's cut off the node
    * either bound tightening failed to identify a possible cutoff due to tolerances
    * or the LP solution that we try to enforce here is not within bounds (see st_e40)
    * TODO if there is a gap left and LP solution is not within bounds, then pass modified LP solution to heur_trysol?
    */
   SCIPdebugMsg(scip, "enforcement with max. violation %g, auxviolation %g, failed; cutting off node\n", maxviol, maxauxviolation);
   *result = SCIP_CUTOFF;
   ++conshdlrdata->ndesperatecutoff;

   return SCIP_OKAY;
}

/** checks for a linear variable that can be increased or decreased without harming feasibility */
static
void consdataFindUnlockedLinearVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONSDATA*        consdata            /**< constraint data */
   )
{
   int poslock;
   int neglock;
   int i;

   assert(conshdlr != NULL);
   assert(consdata != NULL);

   consdata->linvarincr = NULL;
   consdata->linvardecr = NULL;
   consdata->linvarincrcoef = 0.0;
   consdata->linvardecrcoef = 0.0;

   /* root expression is not a sum -> no unlocked linear variable available */
   if( SCIPgetConsExprExprHdlr(consdata->expr) != SCIPgetConsExprExprHdlrSum(conshdlr) )
      return;

   for( i = 0; i < SCIPgetConsExprExprNChildren(consdata->expr); ++i )
   {
      SCIP_CONSEXPR_EXPR* child;

      child = SCIPgetConsExprExprChildren(consdata->expr)[i];
      assert(child != NULL);

      /* check whether the child is a variable expression */
      if( SCIPisConsExprExprVar(child) )
      {
         SCIP_VAR* var = SCIPgetConsExprExprVarVar(child);
         SCIP_Real coef = SCIPgetConsExprExprSumCoefs(consdata->expr)[i];

         if( coef > 0.0 )
         {
            poslock = !SCIPisInfinity(scip,  consdata->rhs) ? 1 : 0;
            neglock = !SCIPisInfinity(scip, -consdata->lhs) ? 1 : 0;
         }
         else
         {
            poslock = !SCIPisInfinity(scip, -consdata->lhs) ? 1 : 0;
            neglock = !SCIPisInfinity(scip,  consdata->rhs) ? 1 : 0;
         }

         if( SCIPvarGetNLocksDownType(var, SCIP_LOCKTYPE_MODEL) - neglock == 0 )
         {
            /* for a*x + f(y) \in [lhs, rhs], we can decrease x without harming other constraints */
            /* if we have already one candidate, then take the one where the loss in the objective function is less */
            if( (consdata->linvardecr == NULL) ||
               (SCIPvarGetObj(consdata->linvardecr) / consdata->linvardecrcoef > SCIPvarGetObj(var) / coef) )
            {
               consdata->linvardecr = var;
               consdata->linvardecrcoef = coef;
            }
         }

         if( SCIPvarGetNLocksUpType(var, SCIP_LOCKTYPE_MODEL) - poslock == 0 )
         {
            /* for a*x + f(y) \in [lhs, rhs], we can increase x without harm */
            /* if we have already one candidate, then take the one where the loss in the objective function is less */
            if( (consdata->linvarincr == NULL) ||
               (SCIPvarGetObj(consdata->linvarincr) / consdata->linvarincrcoef > SCIPvarGetObj(var) / coef) )
            {
               consdata->linvarincr = var;
               consdata->linvarincrcoef = coef;
            }
         }
      }
   }

   assert(consdata->linvarincr == NULL || consdata->linvarincrcoef != 0.0);
   assert(consdata->linvardecr == NULL || consdata->linvardecrcoef != 0.0);

#ifdef SCIP_DEBUG
   if( consdata->linvarincr != NULL )
   {
      SCIPdebugMsg(scip, "may increase <%s> to become feasible\n", SCIPvarGetName(consdata->linvarincr));
   }
   if( consdata->linvardecr != NULL )
   {
      SCIPdebugMsg(scip, "may decrease <%s> to become feasible\n", SCIPvarGetName(consdata->linvardecr));
   }
#endif
}

/** Given a solution where every expression constraint is either feasible or can be made feasible by
 *  moving a linear variable, construct the corresponding feasible solution and pass it to the trysol heuristic.
 *
 *  The method assumes that this is always possible and that not all constraints are feasible already.
 */
static
SCIP_RETCODE proposeFeasibleSolution(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONS**           conss,              /**< constraints to process */
   int                   nconss,             /**< number of constraints */
   SCIP_SOL*             sol,                /**< solution to process */
   SCIP_Bool*            success             /**< buffer to store whether we succeeded to construct a solution that satisfies all provided constraints */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_SOL* newsol;
   int c;

   assert(scip  != NULL);
   assert(conshdlr != NULL);
   assert(conss != NULL || nconss == 0);
   assert(success != NULL);

   *success = FALSE;

   /* don't propose new solutions if not in presolve or solving */
   if( SCIPgetStage(scip) < SCIP_STAGE_INITPRESOLVE || SCIPgetStage(scip) >= SCIP_STAGE_SOLVED )
      return SCIP_OKAY;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   if( sol != NULL )
   {
      SCIP_CALL( SCIPcreateSolCopy(scip, &newsol, sol) );
   }
   else
   {
      SCIP_CALL( SCIPcreateLPSol(scip, &newsol, NULL) );
   }
   SCIP_CALL( SCIPunlinkSol(scip, newsol) );
   SCIPdebugMsg(scip, "attempt to make solution from <%s> feasible by shifting linear variable\n",
      sol != NULL ? (SCIPsolGetHeur(sol) != NULL ? SCIPheurGetName(SCIPsolGetHeur(sol)) : "tree") : "LP");

   for( c = 0; c < nconss; ++c )
   {
      SCIP_CONSDATA* consdata = SCIPconsGetData(conss[c]);  /*lint !e613*/
      SCIP_Real viol = 0.0;
      SCIP_Real delta;
      SCIP_Real gap;

      assert(consdata != NULL);

      /* get absolute violation and sign */
      if( SCIPisGT(scip, consdata->lhsviol, SCIPfeastol(scip)) )
         viol = consdata->lhsviol; /* lhs - activity */
      else if( SCIPisGT(scip, consdata->rhsviol, SCIPfeastol(scip)) )
         viol = -consdata->rhsviol; /* rhs - activity */
      else
         continue; /* constraint is satisfied */

      if( consdata->linvarincr != NULL &&
         ((viol > 0.0 && consdata->linvarincrcoef > 0.0) || (viol < 0.0 && consdata->linvarincrcoef < 0.0)) )
      {
         SCIP_VAR* var = consdata->linvarincr;

         /* compute how much we would like to increase var */
         delta = viol / consdata->linvarincrcoef;
         assert(delta > 0.0);

         /* if var has an upper bound, may need to reduce delta */
         if( !SCIPisInfinity(scip, SCIPvarGetUbGlobal(var)) )
         {
            gap = SCIPvarGetUbGlobal(var) - SCIPgetSolVal(scip, newsol, var);
            delta = MIN(MAX(0.0, gap), delta);
         }
         if( SCIPisPositive(scip, delta) )
         {
            /* if variable is integral, round delta up so that it will still have an integer value */
            if( SCIPvarIsIntegral(var) )
               delta = SCIPceil(scip, delta);

            SCIP_CALL( SCIPincSolVal(scip, newsol, var, delta) );
            SCIPdebugMsg(scip, "increase <%s> by %g to %g to remedy lhs-violation %g of cons <%s>\n",
               SCIPvarGetName(var), delta, SCIPgetSolVal(scip, newsol, var), viol, SCIPconsGetName(conss[c]));  /*lint !e613*/

            /* adjust constraint violation, if satisfied go on to next constraint */
            viol -= consdata->linvarincrcoef * delta;
            if( SCIPisZero(scip, viol) )
               continue;
         }
      }

      assert(viol != 0.0);
      if( consdata->linvardecr != NULL &&
         ((viol > 0.0 && consdata->linvardecrcoef < 0.0) || (viol < 0.0 && consdata->linvardecrcoef > 0.0)) )
      {
         SCIP_VAR* var = consdata->linvardecr;

         /* compute how much we would like to decrease var */
         delta = viol / consdata->linvardecrcoef;
         assert(delta < 0.0);

         /* if var has a lower bound, may need to reduce delta */
         if( !SCIPisInfinity(scip, -SCIPvarGetLbGlobal(var)) )
         {
            gap = SCIPgetSolVal(scip, newsol, var) - SCIPvarGetLbGlobal(var);
            delta = MAX(MIN(0.0, gap), delta);
         }
         if( SCIPisNegative(scip, delta) )
         {
            /* if variable is integral, round delta down so that it will still have an integer value */
            if( SCIPvarIsIntegral(var) )
               delta = SCIPfloor(scip, delta);
            SCIP_CALL( SCIPincSolVal(scip, newsol, consdata->linvardecr, delta) );
            /*lint --e{613} */
            SCIPdebugMsg(scip, "increase <%s> by %g to %g to remedy rhs-violation %g of cons <%s>\n",
               SCIPvarGetName(var), delta, SCIPgetSolVal(scip, newsol, var), viol, SCIPconsGetName(conss[c]));

            /* adjust constraint violation, if satisfied go on to next constraint */
            viol -= consdata->linvardecrcoef * delta;
            if( SCIPisZero(scip, viol) )
               continue;
         }
      }

      /* still here... so probably we could not make constraint feasible due to variable bounds, thus give up */
      break;
   }

   /* if we have a solution that should satisfy all quadratic constraints and has a better objective than the current upper bound,
    * then pass it to the trysol heuristic
    */
   if( c == nconss && (SCIPisInfinity(scip, SCIPgetUpperbound(scip)) || SCIPisSumLT(scip, SCIPgetSolTransObj(scip, newsol), SCIPgetUpperbound(scip))) )
   {
      SCIPdebugMsg(scip, "pass solution with objective val %g to trysol heuristic\n", SCIPgetSolTransObj(scip, newsol));

      assert(conshdlrdata->trysolheur != NULL);
      SCIP_CALL( SCIPheurPassSolTrySol(scip, conshdlrdata->trysolheur, newsol) );

      *success = TRUE;
   }

   SCIP_CALL( SCIPfreeSol(scip, &newsol) );

   return SCIP_OKAY;
}

/** merges constraints that have the same root expression */
static
SCIP_RETCODE presolMergeConss(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           conss,              /**< constraints to process */
   int                   nconss,             /**< number of constraints */
   SCIP_Bool*            success
   )
{
   int i;

   assert(success != NULL);

   *success = FALSE;

   /* iterate through all pairs of constraints */
   for( i = 0; i < nconss - 1; ++i )
   {
      SCIP_CONSDATA* consdata1;
      SCIP_Bool updatedlocks;
      int j;

      /* ignore deleted constraints */
      if( SCIPconsIsDeleted(conss[i]) )
         continue;

      consdata1 = SCIPconsGetData(conss[i]);
      updatedlocks = FALSE;

      for( j = i + 1; j < nconss; ++j )
      {
         SCIP_CONSDATA* consdata2;

         /* ignore deleted constraints */
         if( SCIPconsIsDeleted(conss[j]) )
            continue;

         consdata2 = SCIPconsGetData(conss[j]);

         /* merge constraint sides; delete constraint */
         if( consdata1->expr == consdata2->expr )
         {
            SCIPdebugMsg(scip, "merge constraint %g <= %s <= %g with %g <= %s <= %g\n", consdata1->lhs,
               SCIPconsGetName(conss[i]), consdata1->rhs, consdata2->lhs, SCIPconsGetName(conss[j]), consdata2->rhs);

            /* check whether locks need to be updated */
            if( (SCIPisInfinity(scip, -consdata1->lhs) && !SCIPisInfinity(scip, -consdata2->lhs))
               || (SCIPisInfinity(scip, consdata1->rhs) && !SCIPisInfinity(scip, consdata2->rhs)) )
            {
               SCIP_CALL( addLocks(scip, conss[i], -consdata1->nlockspos, -consdata1->nlocksneg) );
               updatedlocks = TRUE;
            }

            consdata1->lhs = MAX(consdata1->lhs, consdata2->lhs);
            consdata1->rhs = MIN(consdata1->rhs, consdata2->rhs);

            SCIP_CALL( SCIPdelCons(scip, conss[j]) );
            *success = TRUE;
         }
      }

      /* recompute locks */
      if( updatedlocks )
      {
         SCIP_CALL( addLocks(scip, conss[i], consdata1->nlockspos, consdata1->nlocksneg) );
      }
   }

   return SCIP_OKAY;
}

/** print statistics for expression handlers */
static
void printExprHdlrStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   FILE*                 file                /**< file handle, or NULL for standard out */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   int i;

   assert(scip != NULL);
   assert(conshdlr != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIPinfoMessage(scip, file, "Expression Handlers: %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
      "SimplCalls", "Simplified", "SepaCalls", "PropCalls", "Cuts", "Cutoffs", "DomReds", "BranchScor", "SepaTime", "PropTime", "IntEvalTi", "SimplifyTi");

   for( i = 0; i < conshdlrdata->nexprhdlrs; ++i )
   {
      SCIP_CONSEXPR_EXPRHDLR* exprhdlr = conshdlrdata->exprhdlrs[i];
      assert(exprhdlr != NULL);

      SCIPinfoMessage(scip, file, "  %-17s:", exprhdlr->name);
      SCIPinfoMessage(scip, file, " %10lld", exprhdlr->nsimplifycalls);
      SCIPinfoMessage(scip, file, " %10lld", exprhdlr->nsimplified);
      SCIPinfoMessage(scip, file, " %10lld", exprhdlr->nsepacalls);
      SCIPinfoMessage(scip, file, " %10lld", exprhdlr->npropcalls);
      SCIPinfoMessage(scip, file, " %10lld", exprhdlr->ncutsfound);
      SCIPinfoMessage(scip, file, " %10lld", exprhdlr->ncutoffs);
      SCIPinfoMessage(scip, file, " %10lld", exprhdlr->ndomreds);
      SCIPinfoMessage(scip, file, " %10lld", exprhdlr->nbranchscores);
      SCIPinfoMessage(scip, file, " %10.2f", SCIPgetClockTime(scip, exprhdlr->sepatime));
      SCIPinfoMessage(scip, file, " %10.2f", SCIPgetClockTime(scip, exprhdlr->proptime));
      SCIPinfoMessage(scip, file, " %10.2f", SCIPgetClockTime(scip, exprhdlr->intevaltime));
      SCIPinfoMessage(scip, file, " %10.2f", SCIPgetClockTime(scip, exprhdlr->simplifytime));
      SCIPinfoMessage(scip, file, "\n");
   }
}

/** print statistics for nonlinear handlers */
static
void printNlhdlrStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   FILE*                 file                /**< file handle, or NULL for standard out */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   int i;

   assert(scip != NULL);
   assert(conshdlr != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIPinfoMessage(scip, file, "Nlhdlrs            : %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n", "SepaCalls", "PropCalls", "Detects", "Cuts", "Cutoffs", "DomReds", "BranchScor", "Reforms", "DetectTime", "SepaTime", "PropTime", "IntEvalTi", "ReformTi");

   for( i = 0; i < conshdlrdata->nnlhdlrs; ++i )
   {
      SCIP_CONSEXPR_NLHDLR* nlhdlr = conshdlrdata->nlhdlrs[i];
      assert(nlhdlr != NULL);

      /* skip disabled nlhdlr */
      if( !nlhdlr->enabled )
         continue;

      SCIPinfoMessage(scip, file, "  %-17s:", nlhdlr->name);
      SCIPinfoMessage(scip, file, " %10lld", nlhdlr->nsepacalls);
      SCIPinfoMessage(scip, file, " %10lld", nlhdlr->npropcalls);
      SCIPinfoMessage(scip, file, " %10lld", nlhdlr->ndetections);
      SCIPinfoMessage(scip, file, " %10lld", nlhdlr->ncutsfound);
      SCIPinfoMessage(scip, file, " %10lld", nlhdlr->ncutoffs);
      SCIPinfoMessage(scip, file, " %10lld", nlhdlr->ndomreds);
      SCIPinfoMessage(scip, file, " %10lld", nlhdlr->nbranchscores);
      SCIPinfoMessage(scip, file, " %10lld", nlhdlr->nreformulates);
      SCIPinfoMessage(scip, file, " %10.2f", SCIPgetClockTime(scip, nlhdlr->detecttime));
      SCIPinfoMessage(scip, file, " %10.2f", SCIPgetClockTime(scip, nlhdlr->sepatime));
      SCIPinfoMessage(scip, file, " %10.2f", SCIPgetClockTime(scip, nlhdlr->proptime));
      SCIPinfoMessage(scip, file, " %10.2f", SCIPgetClockTime(scip, nlhdlr->intevaltime));
      SCIPinfoMessage(scip, file, " %10.2f", SCIPgetClockTime(scip, nlhdlr->reformulatetime));
      SCIPinfoMessage(scip, file, "\n");
   }
}

/** print statistics for constraint handlers */
static
void printConshdlrStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   FILE*                 file                /**< file handle, or NULL for standard out */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);
   assert(conshdlr != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIPinfoMessage(scip, file, "ConsExpr Enforce   : %10s %10s %10s\n", "DespBranch", "DespCutoff", "ForceLP");
   SCIPinfoMessage(scip, file, "  %-18s", "");
   SCIPinfoMessage(scip, file, " %10lld", conshdlrdata->ndesperatebranch);
   SCIPinfoMessage(scip, file, " %10lld", conshdlrdata->ndesperatecutoff);
   SCIPinfoMessage(scip, file, " %10lld", conshdlrdata->nforcelp);
   SCIPinfoMessage(scip, file, "\n");
   SCIPinfoMessage(scip, file, "ConsExpr Presolve  : %10s\n", "CanonTime");
   SCIPinfoMessage(scip, file, "  %-18s", "");
   SCIPinfoMessage(scip, file, " %10.2f", SCIPgetClockTime(scip, conshdlrdata->canonicalizetime));
   SCIPinfoMessage(scip, file, "\n");
}

/** @} */

/*
 * Callback methods of constraint handler
 */

/** upgrades quadratic constraint to expr constraint */
static
SCIP_DECL_QUADCONSUPGD(quadconsUpgdExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLR* consexprhdlr;
   SCIP_CONSEXPR_EXPR* expr;
   SCIP_CONSEXPR_EXPR* varexpr;
   SCIP_CONSEXPR_EXPR** varexprs;
   SCIP_CONSEXPR_EXPR* prodexpr;
   SCIP_CONSEXPR_EXPR* powexpr;
   SCIP_CONSEXPR_EXPR* twoexprs[2];
   SCIP_QUADVARTERM* quadvarterm;
   SCIP_BILINTERM* bilinterm;
   int pos;
   int i;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(nupgdconss != NULL);
   assert(upgdconss  != NULL);

   *nupgdconss = 0;

   SCIPdebugMsg(scip, "quadconsUpgdExpr called for constraint <%s>\n", SCIPconsGetName(cons));
   SCIPdebugPrintCons(scip, cons, NULL);

   /* no interest in linear constraints */
   if( SCIPgetNQuadVarTermsQuadratic(scip, cons) == 0 )
      return SCIP_OKAY;

   if( upgdconsssize < 1 )
   {
      /* signal that we need more memory */
      *nupgdconss = -1;
      return SCIP_OKAY;
   }

   if( SCIPgetNBilinTermsQuadratic(scip, cons) > 0 )
   {
      /* we will need SCIPfindQuadVarTermQuadratic later, so ensure now that quad var terms are sorted */
      SCIP_CALL( SCIPsortQuadVarTermsQuadratic(scip, cons) );
   }

   consexprhdlr = SCIPfindConshdlr(scip, "expr");
   assert(consexprhdlr != NULL);

   SCIP_CALL( SCIPcreateConsExprExprSum(scip, consexprhdlr, &expr, 0, NULL, NULL, 0.0) );

   /* append linear terms */
   for( i = 0; i < SCIPgetNLinearVarsQuadratic(scip, cons); ++i )
   {
      SCIP_CALL( SCIPcreateConsExprExprVar(scip, consexprhdlr, &varexpr, SCIPgetLinearVarsQuadratic(scip, cons)[i]) );
      SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, expr, varexpr, SCIPgetCoefsLinearVarsQuadratic(scip, cons)[i]) );
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &varexpr) );
   }

   /* array to store variable expression for each quadratic variable */
   SCIP_CALL( SCIPallocBufferArray(scip, &varexprs, SCIPgetNQuadVarTermsQuadratic(scip, cons)) );

   /* create var exprs for quadratic vars; append linear and square part of quadratic terms */
   for( i = 0; i < SCIPgetNQuadVarTermsQuadratic(scip, cons); ++i )
   {
      quadvarterm = &SCIPgetQuadVarTermsQuadratic(scip, cons)[i];

      SCIP_CALL( SCIPcreateConsExprExprVar(scip, consexprhdlr, &varexprs[i], quadvarterm->var) );

      if( quadvarterm->lincoef != 0.0 )
      {
         SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, expr, varexprs[i], quadvarterm->lincoef) );
      }

      if( quadvarterm->sqrcoef != 0.0 )
      {
         SCIP_CALL( SCIPcreateConsExprExprPow(scip, consexprhdlr, &powexpr, varexprs[i], 2.0) );
         SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, expr, powexpr, quadvarterm->sqrcoef) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &powexpr) );
      }
   }

   /* append bilinear terms */
   for( i = 0; i < SCIPgetNBilinTermsQuadratic(scip, cons); ++i)
   {
      bilinterm = &SCIPgetBilinTermsQuadratic(scip, cons)[i];

      SCIP_CALL( SCIPfindQuadVarTermQuadratic(scip, cons, bilinterm->var1, &pos) );
      assert(pos >= 0);
      assert(pos < SCIPgetNQuadVarTermsQuadratic(scip, cons));
      assert(SCIPgetQuadVarTermsQuadratic(scip, cons)[pos].var == bilinterm->var1);
      twoexprs[0] = varexprs[pos];

      SCIP_CALL( SCIPfindQuadVarTermQuadratic(scip, cons, bilinterm->var2, &pos) );
      assert(pos >= 0);
      assert(pos < SCIPgetNQuadVarTermsQuadratic(scip, cons));
      assert(SCIPgetQuadVarTermsQuadratic(scip, cons)[pos].var == bilinterm->var2);
      twoexprs[1] = varexprs[pos];

      SCIP_CALL( SCIPcreateConsExprExprProduct(scip, consexprhdlr, &prodexpr, 2, twoexprs, 1.0) );
      SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, expr, prodexpr, bilinterm->coef) );
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &prodexpr) );
   }

   /* release variable expressions */
   for( i = 0; i < SCIPgetNQuadVarTermsQuadratic(scip, cons); ++i )
   {
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &varexprs[i]) );
   }

   SCIPfreeBufferArray(scip, &varexprs);

   *nupgdconss = 1;
   SCIP_CALL( SCIPcreateConsExpr(scip, upgdconss, SCIPconsGetName(cons),
      expr, SCIPgetLhsQuadratic(scip, cons), SCIPgetRhsQuadratic(scip, cons),
      SCIPconsIsInitial(cons), SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons),
      SCIPconsIsChecked(cons), SCIPconsIsPropagated(cons),  SCIPconsIsLocal(cons),
      SCIPconsIsModifiable(cons), SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons),
      SCIPconsIsStickingAtNode(cons)) );

   SCIPdebugMsg(scip, "created expr constraint:\n");
   SCIPdebugPrintCons(scip, *upgdconss, NULL);

   SCIP_CALL( SCIPreleaseConsExprExpr(scip, &expr) );

   return SCIP_OKAY;
}

/** upgrades nonlinear constraint to expr constraint */
static
SCIP_DECL_NONLINCONSUPGD(nonlinconsUpgdExpr)
{
   SCIP_CONSHDLR* consexprhdlr;
   SCIP_EXPRGRAPH* exprgraph;
   SCIP_EXPRGRAPHNODE* node;
   SCIP_CONSEXPR_EXPR* expr;

   assert(nupgdconss != NULL);
   assert(upgdconss != NULL);

   *nupgdconss = 0;

   exprgraph = SCIPgetExprgraphNonlinear(scip, SCIPconsGetHdlr(cons));
   node = SCIPgetExprgraphNodeNonlinear(scip, cons);

   SCIPdebugMsg(scip, "nonlinconsUpgdExpr called for constraint <%s>\n", SCIPconsGetName(cons));
   SCIPdebugPrintCons(scip, cons, NULL);

   /* no interest in linear constraints */
   if( node == NULL )
      return SCIP_OKAY;

   consexprhdlr = SCIPfindConshdlr(scip, "expr");
   assert(consexprhdlr != NULL);

   /* try to create a cons_expr expression from an expression graph node */
   SCIP_CALL( SCIPcreateConsExprExpr3(scip, consexprhdlr, &expr, exprgraph, node) );

   /* if that didn't work, then because we do not support a certain expression type yet -> no upgrade */
   if( expr == NULL )
      return SCIP_OKAY;

   if( upgdconsssize < 1 )
   {
      /* request larger upgdconss array */
      *nupgdconss = -1;
      return SCIP_OKAY;
   }

   if( SCIPgetNLinearVarsNonlinear(scip, cons) > 0 )
   {
      /* add linear terms */
      SCIP_CONSEXPR_EXPR* varexpr;
      int i;

      /* ensure expr is a sum expression */
      if( SCIPgetConsExprExprHdlr(expr) != SCIPgetConsExprExprHdlrSum(consexprhdlr) )
      {
         SCIP_CONSEXPR_EXPR* sumexpr;

         SCIP_CALL( SCIPcreateConsExprExprSum(scip, consexprhdlr, &sumexpr, 1, &expr, NULL, 0.0) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &expr) );

         expr = sumexpr;
      }

      for( i = 0; i < SCIPgetNLinearVarsNonlinear(scip, cons); ++i )
      {
         SCIP_CALL( SCIPcreateConsExprExprVar(scip, consexprhdlr, &varexpr, SCIPgetLinearVarsNonlinear(scip, cons)[i]) );
         SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, expr, varexpr, SCIPgetLinearCoefsNonlinear(scip, cons)[i]) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &varexpr) );
      }
   }

   *nupgdconss = 1;
   SCIP_CALL( SCIPcreateConsExpr(scip, upgdconss, SCIPconsGetName(cons),
      expr, SCIPgetLhsNonlinear(scip, cons), SCIPgetRhsNonlinear(scip, cons),
      SCIPconsIsInitial(cons), SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons),
      SCIPconsIsChecked(cons), SCIPconsIsPropagated(cons), SCIPconsIsLocal(cons),
      SCIPconsIsModifiable(cons), SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons),
      SCIPconsIsStickingAtNode(cons)) );

   SCIPdebugMsg(scip, "created expr constraint:\n");
   SCIPdebugPrintCons(scip, *upgdconss, NULL);

   SCIP_CALL( SCIPreleaseConsExprExpr(scip, &expr) );

   return SCIP_OKAY;
}

/** copy method for constraint handler plugins (called when SCIP copies plugins) */
static
SCIP_DECL_CONSHDLRCOPY(conshdlrCopyExpr)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(valid != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);

   /* create basic data of constraint handler and include it to scip */
   SCIP_CALL( includeConshdlrExprBasic(scip) );

   /* copy expression and nonlinear handlers */
   SCIP_CALL( copyConshdlrExprExprHdlr(scip, conshdlr, valid) );

   return SCIP_OKAY;
}

/** destructor of constraint handler to free constraint handler data (called when SCIP is exiting) */
static
SCIP_DECL_CONSFREE(consFreeExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr;
   SCIP_CONSEXPR_NLHDLR* nlhdlr;
   int i;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   for( i = 0; i < conshdlrdata->nexprhdlrs; ++i )
   {
      exprhdlr = conshdlrdata->exprhdlrs[i];
      assert(exprhdlr != NULL);

      if( exprhdlr->freehdlr != NULL )
      {
         SCIP_CALL( (*exprhdlr->freehdlr)(scip, conshdlr, exprhdlr, &exprhdlr->data) );
      }

      /* free clocks */
      SCIP_CALL( SCIPfreeClock(scip, &(exprhdlr)->simplifytime) );
      SCIP_CALL( SCIPfreeClock(scip, &(exprhdlr)->intevaltime) );
      SCIP_CALL( SCIPfreeClock(scip, &(exprhdlr)->proptime) );
      SCIP_CALL( SCIPfreeClock(scip, &(exprhdlr)->sepatime) );

      SCIPfreeMemory(scip, &exprhdlr->name);
      SCIPfreeMemoryNull(scip, &exprhdlr->desc);

      SCIPfreeMemory(scip, &exprhdlr);
   }

   SCIPfreeBlockMemoryArray(scip, &conshdlrdata->exprhdlrs, conshdlrdata->exprhdlrssize);

   for( i = 0; i < conshdlrdata->nnlhdlrs; ++i )
   {
      nlhdlr = conshdlrdata->nlhdlrs[i];
      assert(nlhdlr != NULL);

      if( nlhdlr->freehdlrdata != NULL )
      {
         SCIP_CALL( (*nlhdlr->freehdlrdata)(scip, nlhdlr, &nlhdlr->data) );
      }

      /* free clocks */
      SCIP_CALL( SCIPfreeClock(scip, &nlhdlr->detecttime) );
      SCIP_CALL( SCIPfreeClock(scip, &nlhdlr->sepatime) );
      SCIP_CALL( SCIPfreeClock(scip, &nlhdlr->proptime) );
      SCIP_CALL( SCIPfreeClock(scip, &nlhdlr->intevaltime) );
      SCIP_CALL( SCIPfreeClock(scip, &nlhdlr->reformulatetime) );

      SCIPfreeMemory(scip, &nlhdlr->name);
      SCIPfreeMemoryNull(scip, &nlhdlr->desc);

      SCIPfreeMemory(scip, &nlhdlr);
   }

   SCIPfreeBlockMemoryArrayNull(scip, &conshdlrdata->nlhdlrs, conshdlrdata->nlhdlrssize);
   conshdlrdata->nlhdlrssize = 0;

   /* free upgrade functions */
   for( i = 0; i < conshdlrdata->nexprconsupgrades; ++i )
   {
      assert(conshdlrdata->exprconsupgrades[i] != NULL);
      SCIPfreeBlockMemory(scip, &conshdlrdata->exprconsupgrades[i]);  /*lint !e866*/
   }
   SCIPfreeBlockMemoryArrayNull(scip, &conshdlrdata->exprconsupgrades, conshdlrdata->exprconsupgradessize);

   SCIP_CALL( SCIPfreeClock(scip, &conshdlrdata->canonicalizetime) );

   SCIPfreeMemory(scip, &conshdlrdata);
   SCIPconshdlrSetData(conshdlr, NULL);

   return SCIP_OKAY;
}


/** initialization method of constraint handler (called after problem was transformed) */
static
SCIP_DECL_CONSINIT(consInitExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr;
   SCIP_CONSEXPR_NLHDLR* nlhdlr;
   int i;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   for( i = 0; i < nconss; ++i )
   {
      SCIP_CALL( storeVarExprs(scip, conshdlr, SCIPconsGetData(conss[i])) );
      SCIP_CALL( catchVarEvents(scip, conshdlrdata->eventhdlr, conss[i]) );
   }

   /* sort nonlinear handlers by priority, in decreasing order */
   if( conshdlrdata->nnlhdlrs > 1 )
      SCIPsortDownPtr((void**)conshdlrdata->nlhdlrs, nlhdlrCmp, conshdlrdata->nnlhdlrs);

   /* get heuristics for later use */
   conshdlrdata->subnlpheur = SCIPfindHeur(scip, "subnlp");
   conshdlrdata->trysolheur = SCIPfindHeur(scip, "trysol");

   /* reset statistics in expression handlers */
   for( i = 0; i < conshdlrdata->nexprhdlrs; ++i )
   {
      exprhdlr = conshdlrdata->exprhdlrs[i];
      assert(exprhdlr != NULL);

      exprhdlr->nsepacalls = 0;
      exprhdlr->npropcalls = 0;
      exprhdlr->ncutsfound = 0;
      exprhdlr->ncutoffs = 0;
      exprhdlr->ndomreds = 0;
      exprhdlr->nbranchscores = 0;
      exprhdlr->nsimplifycalls = 0;
      exprhdlr->nsimplified = 0;

      SCIP_CALL( SCIPresetClock(scip, exprhdlr->sepatime) );
      SCIP_CALL( SCIPresetClock(scip, exprhdlr->proptime) );
      SCIP_CALL( SCIPresetClock(scip, exprhdlr->intevaltime) );
      SCIP_CALL( SCIPresetClock(scip, exprhdlr->simplifytime) );
   }

   /* reset statistics in nonlinear handlers */
   for( i = 0; i < conshdlrdata->nnlhdlrs; ++i )
   {
      nlhdlr = conshdlrdata->nlhdlrs[i];
      assert(nlhdlr != NULL);

      nlhdlr->nsepacalls = 0;
      nlhdlr->npropcalls = 0;
      nlhdlr->ncutsfound = 0;
      nlhdlr->ncutoffs = 0;
      nlhdlr->ndomreds = 0;
      nlhdlr->nbranchscores = 0;
      nlhdlr->ndetections = 0;

      SCIP_CALL( SCIPresetClock(scip, nlhdlr->detecttime) );
      SCIP_CALL( SCIPresetClock(scip, nlhdlr->sepatime) );
      SCIP_CALL( SCIPresetClock(scip, nlhdlr->proptime) );
      SCIP_CALL( SCIPresetClock(scip, nlhdlr->intevaltime) );
      SCIP_CALL( SCIPresetClock(scip, nlhdlr->reformulatetime) );
   }

   /* reset statistics in constraint handler */
   conshdlrdata->ndesperatebranch = 0;
   conshdlrdata->ndesperatecutoff = 0;
   conshdlrdata->nforcelp = 0;
   SCIP_CALL( SCIPresetClock(scip, conshdlrdata->canonicalizetime) );

   return SCIP_OKAY;
}


/** deinitialization method of constraint handler (called before transformed problem is freed) */
static
SCIP_DECL_CONSEXIT(consExitExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   int i;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   for( i = 0; i < nconss; ++i )
   {
      SCIP_CALL( dropVarEvents(scip, conshdlrdata->eventhdlr, conss[i]) );
      SCIP_CALL( freeVarExprs(scip, SCIPconsGetData(conss[i])) );
   }

   conshdlrdata->subnlpheur = NULL;
   conshdlrdata->trysolheur = NULL;

   return SCIP_OKAY;
}


/** presolving initialization method of constraint handler (called when presolving is about to begin) */
static
SCIP_DECL_CONSINITPRE(consInitpreExpr)
{  /*lint --e{715}*/

   /* remove auxiliary variables when a restart has happened; this ensures that the previous branch-and-bound tree
    * removed all of his captures on variables; variables that are not release by any plug-in (nuses = 2) will then
    * unlocked and freed
    */
   if( SCIPgetNRuns(scip) > 1 )
   {
      SCIP_CALL( freeAuxVars(scip, conshdlr, conss, nconss) );
   }

   return SCIP_OKAY;
}


/** presolving deinitialization method of constraint handler (called after presolving has been finished) */
static
SCIP_DECL_CONSEXITPRE(consExitpreExpr)
{  /*lint --e{715}*/

   if( nconss > 0 )
   {
      SCIP_Bool infeasible;
      int i;

      /* simplify constraints and replace common subexpressions */
      SCIP_CALL( canonicalizeConstraints(scip, conshdlr, conss, nconss, &infeasible) );
      /* currently SCIP does not offer to communicate this,
       * but at the moment this can only become true if canonicalizeConstraints called detectNlhdlrs, which it doesn't do in EXITPRESOLVE stage
       */
      assert(!infeasible);

      /* call curvature detection of expression handlers */
      for( i = 0; i < nconss; ++i )
      {
         SCIP_CONSDATA* consdata = SCIPconsGetData(conss[i]);
         assert(consdata != NULL);
         assert(consdata->expr != NULL);

         /* evaluate all expressions for curvature check */
         SCIP_CALL( SCIPcomputeConsExprExprCurvature(scip, consdata->expr) );
      }

      /* tell SCIP that we have something nonlinear */
      SCIPenableNLP(scip);
   }

   return SCIP_OKAY;
}


/** solving process initialization method of constraint handler (called when branch and bound process is about to begin) */
static
SCIP_DECL_CONSINITSOL(consInitsolExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   int c;
   int i;

   for( c = 0; c < nconss; ++c )
   {
      consdata = SCIPconsGetData(conss[c]);  /*lint !e613*/
      assert(consdata != NULL);

      /* check for a linear variable that can be increase or decreased without harming feasibility */
      consdataFindUnlockedLinearVar(scip, conshdlr, consdata);

      /* add nlrow representation to NLP, if NLP had been constructed */
      if( SCIPisNLPConstructed(scip) && SCIPconsIsEnabled(conss[c]) )
      {
         if( consdata->nlrow == NULL )
         {
            SCIP_CALL( createNlRow(scip, conss[c]) );
            assert(consdata->nlrow != NULL);
         }
         SCIP_CALL( SCIPaddNlRow(scip, consdata->nlrow) );
      }
   }

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* initialize nonlinear handlers */
   for( i = 0; i < conshdlrdata->nnlhdlrs; ++i )
   {
      SCIP_CONSEXPR_NLHDLR* nlhdlr;

      nlhdlr = conshdlrdata->nlhdlrs[i];
      if( nlhdlr->init != NULL )
      {
         SCIP_CALL( (*nlhdlr->init)(scip, nlhdlr) );
      }
   }

   return SCIP_OKAY;
}

/** solving process deinitialization method of constraint handler (called before branch and bound process data is freed) */
static
SCIP_DECL_CONSEXITSOL(consExitsolExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSEXPR_EXPR* expr;
   int c;
   int i;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, NULL, SCIP_CONSEXPRITERATOR_DFS, FALSE) );

   /* call deinitialization callbacks of expression and nonlinear handlers
    * free nonlinear handlers information from expressions
    * remove auxiliary variables from expressions, if not restarting; otherwise do so in CONSINITPRE
    */
   for( c = 0; c < nconss; ++c )
   {
      assert(conss != NULL);
      assert(conss[c] != NULL);

      consdata = SCIPconsGetData(conss[c]);
      assert(consdata != NULL);

      for( expr = SCIPexpriteratorRestartDFS(it, consdata->expr); !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
      {
         SCIPdebugMsg(scip, "exitsepa and free nonlinear handler data for expression %p\n", (void*)expr);

         /* remove nonlinear handlers in expression and their data and auxiliary variables if not restarting */
         SCIP_CALL( freeEnfoData(scip, expr, !restart) );
      }
   }

   SCIPexpriteratorFree(&it);

   /* deinitialize nonlinear handlers */
   for( i = 0; i < conshdlrdata->nnlhdlrs; ++i )
   {
      SCIP_CONSEXPR_NLHDLR* nlhdlr;

      nlhdlr = conshdlrdata->nlhdlrs[i];
      if( nlhdlr->exit != NULL )
      {
         SCIP_CALL( (*nlhdlr->exit)(scip, nlhdlr) );
      }
   }

   /* free nonlinear row representations */
   for( c = 0; c < nconss; ++c )
   {
      consdata = SCIPconsGetData(conss[c]);  /*lint !e613*/
      assert(consdata != NULL);

      if( consdata->nlrow != NULL )
      {
         SCIP_CALL( SCIPreleaseNlRow(scip, &consdata->nlrow) );
      }
   }

   return SCIP_OKAY;
}


/** frees specific constraint data */
static
SCIP_DECL_CONSDELETE(consDeleteExpr)
{  /*lint --e{715}*/
   assert(consdata != NULL);
   assert(*consdata != NULL);
   assert((*consdata)->expr != NULL);
   assert((*consdata)->nvarexprs == 0);
   assert((*consdata)->varexprs == NULL);

   /* constraint locks should have been removed */
   assert((*consdata)->nlockspos == 0);
   assert((*consdata)->nlocksneg == 0);

   SCIP_CALL( SCIPreleaseConsExprExpr(scip, &(*consdata)->expr) );

   /* free nonlinear row representation */
   if( (*consdata)->nlrow != NULL )
   {
      SCIP_CALL( SCIPreleaseNlRow(scip, &(*consdata)->nlrow) );
   }

   SCIPfreeBlockMemory(scip, consdata);

   return SCIP_OKAY;
}


/** transforms constraint data into data belonging to the transformed problem */
static
SCIP_DECL_CONSTRANS(consTransExpr)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* targetexpr;
   SCIP_CONSDATA* sourcedata;

   sourcedata = SCIPconsGetData(sourcecons);
   assert(sourcedata != NULL);

   /* get a copy of sourceexpr with transformed vars */
   SCIP_CALL( copyExpr(scip, scip, conshdlr, sourcedata->expr, &targetexpr, transformVar, NULL) );

   if( targetexpr == NULL )
   {
      SCIPerrorMessage("Copying expression in consTransExpr failed.\n");
      return SCIP_ERROR;
   }

   /* create transformed cons (captures targetexpr) */
   SCIP_CALL( SCIPcreateConsExpr(scip, targetcons, SCIPconsGetName(sourcecons),
      targetexpr, sourcedata->lhs, sourcedata->rhs,
      SCIPconsIsInitial(sourcecons), SCIPconsIsSeparated(sourcecons), SCIPconsIsEnforced(sourcecons),
      SCIPconsIsChecked(sourcecons), SCIPconsIsPropagated(sourcecons),
      SCIPconsIsLocal(sourcecons), SCIPconsIsModifiable(sourcecons),
      SCIPconsIsDynamic(sourcecons), SCIPconsIsRemovable(sourcecons), SCIPconsIsStickingAtNode(sourcecons)) );

   /* release target expr */
   SCIP_CALL( SCIPreleaseConsExprExpr(scip, &targetexpr) );

   return SCIP_OKAY;
}


/** LP initialization method of constraint handler (called before the initial LP relaxation at a node is solved) */
static
SCIP_DECL_CONSINITLP(consInitlpExpr)
{
   /* register non linear handlers TODO: do we want this here? */
   SCIP_CALL( detectNlhdlrs(scip, conshdlr, conss, nconss, infeasible) );

   /* if creating auxiliary variables detected an infeasible (because of bounds), stop initing lp */
   if( *infeasible )
      return SCIP_OKAY;

   /* call seaparation initialization callbacks of the expression handlers */
   SCIP_CALL( initSepa(scip, conshdlr, conss, nconss, infeasible) );

   return SCIP_OKAY;
}


/** separation method of constraint handler for LP solutions */
static
SCIP_DECL_CONSSEPALP(consSepalpExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   unsigned int soltag;
   int c;

   *result = SCIP_DIDNOTFIND;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   soltag = ++conshdlrdata->lastsoltag;

   /* compute violations */
   for( c = 0; c < nconss; ++c )
   {
      assert(conss[c] != NULL);
      SCIP_CALL( computeViolation(scip, conss[c], NULL, soltag) );
   }

   /* call separation
    * TODO revise minviolation, should it be larger than feastol?
    */
   SCIP_CALL( separatePoint(scip, conshdlr, conss, nconss, nusefulconss, NULL, soltag, SCIPfeastol(scip), SCIPgetSepaMinEfficacy(scip), result, NULL) );

   return SCIP_OKAY;
}


/** separation method of constraint handler for arbitrary primal solutions */
static
SCIP_DECL_CONSSEPASOL(consSepasolExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   unsigned int soltag;
   int c;

   *result = SCIP_DIDNOTFIND;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   soltag = ++conshdlrdata->lastsoltag;

   /* compute violations */
   for( c = 0; c < nconss; ++c )
   {
      assert(conss[c] != NULL);
      SCIP_CALL( computeViolation(scip, conss[c], sol, soltag) );
   }

   /* call separation
    * TODO revise minviolation, should it be larger than feastol?
    */
   SCIP_CALL( separatePoint(scip, conshdlr, conss, nconss, nusefulconss, sol, soltag, SCIPfeastol(scip), SCIPgetSepaMinEfficacy(scip), result, NULL) );

   return SCIP_OKAY;
}


/** constraint enforcing method of constraint handler for LP solutions */
static
SCIP_DECL_CONSENFOLP(consEnfolpExpr)
{  /*lint --e{715}*/
   SCIP_CALL( enforceConstraints(scip, conshdlr, conss, nconss, nusefulconss, NULL, result) );

   return SCIP_OKAY;
}

/** constraint enforcing method of constraint handler for relaxation solutions */
static
SCIP_DECL_CONSENFORELAX(consEnforelaxExpr)
{  /*lint --e{715}*/
   SCIP_CALL( enforceConstraints(scip, conshdlr, conss, nconss, nusefulconss, sol, result) );

   return SCIP_OKAY;
}

/** constraint enforcing method of constraint handler for pseudo solutions */
static
SCIP_DECL_CONSENFOPS(consEnfopsExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata = SCIPconshdlrGetData(conshdlr);
   SCIP_CONSDATA* consdata;
   SCIP_RESULT propresult;
   SCIP_Bool force;
   unsigned int soltag;
   int nchgbds;
   int ndelconss;
   int nnotify;
   int c;

   /* TODO call enforceConstraints here, maybe with some flag to indicate ENFOPS? */

   /* force tightenings when calling enforcement for the first time for a node */
   force = conshdlrdata->lastenfopsnodenum == SCIPnodeGetNumber(SCIPgetCurrentNode(scip));
   conshdlrdata->lastenfopsnodenum = SCIPnodeGetNumber(SCIPgetCurrentNode(scip));

   soltag = ++conshdlrdata->lastsoltag;

   *result = SCIP_FEASIBLE;
   for( c = 0; c < nconss; ++c )
   {
      SCIP_CALL( computeViolation(scip, conss[c], NULL, soltag) );

      consdata = SCIPconsGetData(conss[c]);
      if( SCIPisGT(scip, MAX(consdata->lhsviol, consdata->rhsviol), SCIPfeastol(scip)) )
      {
         *result = SCIP_INFEASIBLE;
         break;
      }
   }

   if( *result == SCIP_FEASIBLE )
      return SCIP_OKAY;

   /* try to propagate */
   nchgbds = 0;
   ndelconss = 0;
   SCIP_CALL( propConss(scip, conshdlr, conss, nconss, force, &propresult, &nchgbds, &ndelconss) );

   if( (propresult == SCIP_CUTOFF) || (propresult == SCIP_REDUCEDDOM) )
   {
      *result = propresult;
      return SCIP_OKAY;
   }

   /* find branching candidates */
   SCIP_CALL( registerBranchingCandidates(scip, conshdlr, conss, nconss, NULL, soltag, SCIPfeastol(scip), TRUE, &nnotify) );
   if( nnotify > 0 )
   {
      SCIPdebugMsg(scip, "registered %d external branching candidates\n", nnotify);

      return SCIP_OKAY;
   }

   /* TODO try registerBranchingCandidatesAllUnfixed ? */

   SCIPdebugMsg(scip, "could not find branching candidates, forcing to solve LP\n");
   *result = SCIP_SOLVELP;
   ++conshdlrdata->nforcelp;

   return SCIP_OKAY;
}


/** feasibility check method of constraint handler for integral solutions */
static
SCIP_DECL_CONSCHECK(consCheckExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA*     consdata;
   SCIP_Real          maxviol;
   SCIP_Bool          maypropfeasible;
   unsigned int soltag;
   int c;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(conss != NULL || nconss == 0);
   assert(result != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   *result = SCIP_FEASIBLE;
   soltag = ++(conshdlrdata->lastsoltag);
   maxviol = 0.0;
   maypropfeasible = conshdlrdata->trysolheur != NULL && SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED
      && SCIPgetStage(scip) <= SCIP_STAGE_SOLVING;

   /* check nonlinear constraints for feasibility */
   for( c = 0; c < nconss; ++c )
   {
      assert(conss != NULL && conss[c] != NULL);
      SCIP_CALL( computeViolation(scip, conss[c], sol, soltag) );

      consdata = SCIPconsGetData(conss[c]);
      assert(consdata != NULL);

      if( SCIPisGT(scip, consdata->lhsviol, SCIPfeastol(scip)) || SCIPisGT(scip, consdata->rhsviol, SCIPfeastol(scip)) )
      {
         *result = SCIP_INFEASIBLE;
         maxviol = MAX3(maxviol, consdata->lhsviol, consdata->rhsviol);

         /* print reason for infeasibility */
         if( printreason )
         {
            SCIP_CALL( SCIPprintCons(scip, conss[c], NULL) );
            SCIPinfoMessage(scip, NULL, ";\n");

            if( SCIPisGT(scip, consdata->lhsviol, SCIPfeastol(scip)) )
            {
               SCIPinfoMessage(scip, NULL, "violation: left hand side is violated by %.15g\n", consdata->lhsviol);
            }
            if( SCIPisGT(scip, consdata->rhsviol, SCIPfeastol(scip)) )
            {
               SCIPinfoMessage(scip, NULL, "violation: right hand side is violated by %.15g\n", consdata->rhsviol);
            }
         }
         else if( (conshdlrdata->subnlpheur == NULL || sol == NULL) && !maypropfeasible && !completely )
         {
            /* if we don't want to pass to subnlp heuristic and don't need to print reasons, then can stop checking here */
            return SCIP_OKAY;
         }

         /* do not try to shift linear variables if violation is at infinity (leads to setting variable to infinity in solution, which is not allowed) */
         if( maypropfeasible && (SCIPisInfinity(scip, consdata->lhsviol) || SCIPisInfinity(scip, consdata->rhsviol)) )
            maypropfeasible = FALSE;

         if( maypropfeasible )
         {
            /* update information on linear variables that may be in- or decreased, if initsolve has not done so yet */
            if( SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED && SCIPgetStage(scip) < SCIP_STAGE_INITSOLVE )
               consdataFindUnlockedLinearVar(scip, conshdlr, consdata);

            if( SCIPisGT(scip, consdata->lhsviol, SCIPfeastol(scip)) )
            {
               /* check if there is a variable which may help to get the left hand side satisfied
                * if there is no such variable, then we cannot get feasible
                */
               if( !(consdata->linvarincr != NULL && consdata->linvarincrcoef > 0.0) &&
                  !(consdata->linvardecr != NULL && consdata->linvardecrcoef < 0.0) )
                  maypropfeasible = FALSE;
            }
            else
            {
               assert(SCIPisGT(scip, consdata->rhsviol, SCIPfeastol(scip)));
               /* check if there is a variable which may help to get the right hand side satisfied
                * if there is no such variable, then we cannot get feasible
                */
               if( !(consdata->linvarincr != NULL && consdata->linvarincrcoef < 0.0) &&
                  !(consdata->linvardecr != NULL && consdata->linvardecrcoef > 0.0) )
                  maypropfeasible = FALSE;
            }
         }
      }
   }

   if( *result == SCIP_INFEASIBLE && maypropfeasible )
   {
      SCIP_Bool success;

      SCIP_CALL( proposeFeasibleSolution(scip, conshdlr, conss, nconss, sol, &success) );

      /* do not pass solution to NLP heuristic if we made it feasible this way */
      if( success )
         return SCIP_OKAY;
   }

   if( *result == SCIP_INFEASIBLE && conshdlrdata->subnlpheur != NULL && sol != NULL && !SCIPisInfinity(scip, maxviol) )
   {
      SCIP_CALL( SCIPupdateStartpointHeurSubNlp(scip, conshdlrdata->subnlpheur, sol, maxviol) );
   }

   return SCIP_OKAY;
}


/** domain propagation method of constraint handler */
static
SCIP_DECL_CONSPROP(consPropExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata = SCIPconshdlrGetData(conshdlr);
   SCIP_Bool force;
   int nchgbds;
   int ndelconss;

   /* force tightenings when calling propagation for the first time for a node */
   force = conshdlrdata->lastpropnodenum != SCIPnodeGetNumber(SCIPgetCurrentNode(scip));
   conshdlrdata->lastpropnodenum = SCIPnodeGetNumber(SCIPgetCurrentNode(scip));

   nchgbds = 0;
   ndelconss = 0;

   SCIP_CALL( propConss(scip, conshdlr, conss, nconss, force, result, &nchgbds, &ndelconss) );
   assert(nchgbds >= 0);

   /* TODO would it make sense to check for redundant constraints? */

   return SCIP_OKAY;
}


/** presolving method of constraint handler */
static
SCIP_DECL_CONSPRESOL(consPresolExpr)
{  /*lint --e{715}*/
   SCIP_Bool infeasible;
   int c;

   *result = SCIP_DIDNOTFIND;

   /* simplify constraints and replace common subexpressions */
   SCIP_CALL( canonicalizeConstraints(scip, conshdlr, conss, nconss, &infeasible) );
   if( infeasible )
   {
      *result = SCIP_CUTOFF;
      return SCIP_OKAY;
   }

   /* merge constraints with the same root expression */
   if( (presoltiming & SCIP_PRESOLTIMING_EXHAUSTIVE) != 0 )
   {
      SCIP_Bool success;

      SCIP_CALL( presolMergeConss(scip, conss, nconss, &success) );
      if( success )
         *result = SCIP_SUCCESS;
   }

   /* propagate constraints */
   SCIP_CALL( propConss(scip, conshdlr, conss, nconss, FALSE, result, nchgbds, ndelconss) );
   if( *result == SCIP_CUTOFF )
      return SCIP_OKAY;

   /* check for redundant constraints, remove constraints that are a value expression */
   SCIP_CALL( checkRedundancyConss(scip, conshdlr, conss, nconss, &infeasible, ndelconss, nchgbds) );
   if( infeasible )
   {
      *result = SCIP_CUTOFF;
      return SCIP_OKAY;
   }

   /* try to upgrade constraints */
   for( c = 0; c < nconss; ++c )
   {
      SCIP_Bool upgraded;

      /* skip inactive and deleted constraints */
      if( SCIPconsIsDeleted(conss[c]) || !SCIPconsIsActive(conss[c]) )
         continue;

      SCIP_CALL( presolveUpgrade(scip, conshdlr, conss[c], &upgraded, nupgdconss, naddconss) );  /*lint !e794*/
   }

   if( *ndelconss > 0 || *nchgbds > 0 || *nupgdconss > 0 )
      *result = SCIP_SUCCESS;
   else
      *result = SCIP_DIDNOTFIND;

   return SCIP_OKAY;
}


/** propagation conflict resolving method of constraint handler */
#if 0
static
SCIP_DECL_CONSRESPROP(consRespropExpr)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of expr constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consRespropExpr NULL
#endif


/** variable rounding lock method of constraint handler */
static
SCIP_DECL_CONSLOCK(consLockExpr)
{  /*lint --e{715}*/
   SCIP_CONSDATA* consdata;

   assert(conshdlr != NULL);
   assert(cons != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   if( consdata->expr == NULL )
      return SCIP_OKAY;

   /* add locks */
   SCIP_CALL( addLocks(scip, cons, nlockspos, nlocksneg) );

   return SCIP_OKAY;
}


/** constraint activation notification method of constraint handler */
static
SCIP_DECL_CONSACTIVE(consActiveExpr)
{  /*lint --e{715}*/

   if( SCIPgetStage(scip) > SCIP_STAGE_TRANSFORMED )
   {
      SCIP_CALL( storeVarExprs(scip, conshdlr, SCIPconsGetData(cons)) );
   }

   return SCIP_OKAY;
}


/** constraint deactivation notification method of constraint handler */
static
SCIP_DECL_CONSDEACTIVE(consDeactiveExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   if( SCIPgetStage(scip) > SCIP_STAGE_TRANSFORMED )
   {
      SCIP_CALL( dropVarEvents(scip, conshdlrdata->eventhdlr, cons) );
      SCIP_CALL( freeVarExprs(scip, SCIPconsGetData(cons)) );
   }

   return SCIP_OKAY;
}

/** constraint enabling notification method of constraint handler */
static
SCIP_DECL_CONSENABLE(consEnableExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   if( SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED )
   {
      SCIP_CALL( catchVarEvents(scip, conshdlrdata->eventhdlr, cons) );
   }

   return SCIP_OKAY;
}

/** constraint disabling notification method of constraint handler */
static
SCIP_DECL_CONSDISABLE(consDisableExpr)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   if( SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED )
   {
      SCIP_CALL( dropVarEvents(scip, conshdlrdata->eventhdlr, cons) );
   }

   return SCIP_OKAY;
}

/** variable deletion of constraint handler */
#if 0
static
SCIP_DECL_CONSDELVARS(consDelvarsExpr)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of expr constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consDelvarsExpr NULL
#endif


/** constraint display method of constraint handler */
static
SCIP_DECL_CONSPRINT(consPrintExpr)
{  /*lint --e{715}*/

   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* print left hand side for ranged constraints */
   if( !SCIPisInfinity(scip, -consdata->lhs)
      && !SCIPisInfinity(scip, consdata->rhs)
      && !SCIPisEQ(scip, consdata->lhs, consdata->rhs) )
      SCIPinfoMessage(scip, file, "%.15g <= ", consdata->lhs);

   /* print expression */
   if( consdata->expr != NULL )
   {
      SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, consdata->expr, file) );
   }
   else
   {
      SCIPinfoMessage(scip, file, "0");
   }

   /* print right hand side */
   if( SCIPisEQ(scip, consdata->lhs, consdata->rhs) )
      SCIPinfoMessage(scip, file, " == %.15g", consdata->rhs);
   else if( !SCIPisInfinity(scip, consdata->rhs) )
      SCIPinfoMessage(scip, file, " <= %.15g", consdata->rhs);
   else if( !SCIPisInfinity(scip, -consdata->lhs) )
      SCIPinfoMessage(scip, file, " >= %.15g", consdata->lhs);
   else
      SCIPinfoMessage(scip, file, " [free]");

   return SCIP_OKAY;
}


/** constraint copying method of constraint handler */
static
SCIP_DECL_CONSCOPY(consCopyExpr)
{  /*lint --e{715}*/
   COPY_MAPVAR_DATA mapvardata;
   SCIP_CONSEXPR_EXPR* targetexpr;
   SCIP_CONSDATA* sourcedata;

   assert(cons != NULL);

   sourcedata = SCIPconsGetData(sourcecons);
   assert(sourcedata != NULL);

   mapvardata.varmap = varmap;
   mapvardata.consmap = consmap;
   mapvardata.global = global;
   mapvardata.valid = TRUE; /* hope the best */

   /* get a copy of sourceexpr with transformed vars */
   SCIP_CALL( copyExpr(sourcescip, scip, sourceconshdlr, sourcedata->expr, &targetexpr, copyVar, &mapvardata) );

   if( targetexpr == NULL )
   {
      *cons = NULL;
      *valid = FALSE;

      return SCIP_OKAY;
   }

   /* validity depends only on the SCIPgetVarCopy() returns from copyVar, which are accumulated in mapvardata.valid */
   *valid = mapvardata.valid;

   /* create copy (captures targetexpr) */
   SCIP_CALL( SCIPcreateConsExpr(scip, cons, name != NULL ? name : SCIPconsGetName(sourcecons),
      targetexpr, sourcedata->lhs, sourcedata->rhs,
      initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode) );

   /* release target expr */
   SCIP_CALL( SCIPreleaseConsExprExpr(scip, &targetexpr) );

   return SCIP_OKAY;
}


/** constraint parsing method of constraint handler */
static
SCIP_DECL_CONSPARSE(consParseExpr)
{  /*lint --e{715}*/
   SCIP_Real  lhs;
   SCIP_Real  rhs;
   const char* endptr;
   SCIP_CONSEXPR_EXPR* consexprtree;

   SCIPdebugMsg(scip, "cons_expr::consparse parsing %s\n",str);

   assert(scip != NULL);
   assert(success != NULL);
   assert(str != NULL);
   assert(name != NULL);
   assert(cons != NULL);

   *success = FALSE;

   /* return if string empty */
   if( !*str )
      return SCIP_OKAY;

   endptr = str;

   /* set left and right hand side to their default values */
   lhs = -SCIPinfinity(scip);
   rhs =  SCIPinfinity(scip);

   /* parse constraint to get lhs, rhs, and expression in between (from cons_linear.c::consparse, but parsing whole string first, then getting expression) */

   /* check for left hand side */
   if( isdigit((unsigned char)str[0]) || ((str[0] == '-' || str[0] == '+') && isdigit((unsigned char)str[1])) )
   {
      /* there is a number coming, maybe it is a left-hand-side */
      if( !SCIPstrToRealValue(str, &lhs, (char**)&endptr) )
      {
         SCIPerrorMessage("error parsing number from <%s>\n", str);
         return SCIP_READERROR;
      }

      /* ignore whitespace */
      while( isspace((unsigned char)*endptr) )
         ++endptr;

      if( endptr[0] != '<' || endptr[1] != '=' )
      {
         /* no '<=' coming, so it was the beginning of the expression and not a left-hand-side */
         lhs = -SCIPinfinity(scip);
      }
      else
      {
         /* it was indeed a left-hand-side, so continue parsing after it */
         str = endptr + 2;

         /* ignore whitespace */
         while( isspace((unsigned char)*str) )
            ++str;
      }
   }

   debugParse("str should start at beginning of expr: %s\n", str); /*lint !e506 !e681*/

   /* parse expression: so far we did not allocate memory, so can just return in case of readerror */
   SCIP_CALL( SCIPparseConsExprExpr(scip, conshdlr, str, &str, &consexprtree) );

   /* check for left or right hand side */
   while( isspace((unsigned char)*str) )
      ++str;

   /* check for free constraint */
   if( strncmp(str, "[free]", 6) == 0 )
   {
      if( !SCIPisInfinity(scip, -lhs) )
      {
         SCIPerrorMessage("cannot have left hand side and [free] status \n");
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &consexprtree) );
         return SCIP_OKAY;
      }
      *success = TRUE;
   }
   else
   {
      switch( *str )
      {
         case '<':
            *success = SCIPstrToRealValue(str+2, &rhs, (char**)&endptr);
            break;
         case '=':
            if( !SCIPisInfinity(scip, -lhs) )
            {
               SCIPerrorMessage("cannot have == on rhs if there was a <= on lhs\n");
               SCIP_CALL( SCIPreleaseConsExprExpr(scip, &consexprtree) );
               return SCIP_OKAY;
            }
            else
            {
               *success = SCIPstrToRealValue(str+2, &rhs, (char**)&endptr);
               lhs = rhs;
            }
            break;
         case '>':
            if( !SCIPisInfinity(scip, -lhs) )
            {
               SCIPerrorMessage("cannot have => on rhs if there was a <= on lhs\n");
               SCIP_CALL( SCIPreleaseConsExprExpr(scip, &consexprtree) );
               return SCIP_OKAY;
            }
            else
            {
               *success = SCIPstrToRealValue(str+2, &lhs, (char**)&endptr);
               break;
            }
         case '\0':
            *success = TRUE;
            break;
         default:
            SCIPerrorMessage("unexpected character %c\n", *str);
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &consexprtree) );
            return SCIP_OKAY;
      }
   }

   /* create constraint */
   SCIP_CALL( SCIPcreateConsExpr(scip, cons, name,
      consexprtree, lhs, rhs,
      initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode) );
   assert(*cons != NULL);

   SCIP_CALL( SCIPreleaseConsExprExpr(scip, &consexprtree) );

   debugParse("created expression constraint: <%s>\n", SCIPconsGetName(*cons)); /*lint !e506 !e681*/

   return SCIP_OKAY;
}


/** constraint method of constraint handler which returns the variables (if possible) */
#if 0 /* TODO */
static
SCIP_DECL_CONSGETVARS(consGetVarsExpr)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of expr constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consGetVarsExpr NULL
#endif

/** constraint method of constraint handler which returns the number of variables (if possible) */
#if 0 /* TODO */
static
SCIP_DECL_CONSGETNVARS(consGetNVarsExpr)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of expr constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consGetNVarsExpr NULL
#endif

/** constraint handler method to suggest dive bound changes during the generic diving algorithm */
#if 0 /* TODO? */
static
SCIP_DECL_CONSGETDIVEBDCHGS(consGetDiveBdChgsExpr)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of expr constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consGetDiveBdChgsExpr NULL
#endif

/** output method of statistics table to output file stream 'file' */
static
SCIP_DECL_TABLEOUTPUT(tableOutputExpr)
{ /*lint --e{715}*/
   SCIP_CONSHDLR* conshdlr;

   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   /* print statistics for expression handlers */
   printExprHdlrStatistics(scip, conshdlr, file);

   /* print statistics for nonlinear handlers */
   printNlhdlrStatistics(scip, conshdlr, file);

   /* print statistics for constraint handler */
   printConshdlrStatistics(scip, conshdlr, file);

   return SCIP_OKAY;
}

/** creates the handler for an expression handler and includes it into the expression constraint handler */
SCIP_RETCODE SCIPincludeConsExprExprHdlrBasic(
   SCIP*                       scip,         /**< SCIP data structure */
   SCIP_CONSHDLR*              conshdlr,     /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR**    exprhdlr,     /**< buffer where to store expression handler */
   const char*                 name,         /**< name of expression handler (must not be NULL) */
   const char*                 desc,         /**< description of expression handler (can be NULL) */
   unsigned int                precedence,   /**< precedence of expression operation (used for printing) */
   SCIP_DECL_CONSEXPR_EXPREVAL((*eval)),     /**< point evaluation callback (can not be NULL) */
   SCIP_CONSEXPR_EXPRHDLRDATA* data          /**< data of expression handler (can be NULL) */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(name != NULL);
   assert(exprhdlr != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIP_CALL( SCIPallocClearMemory(scip, exprhdlr) );

   SCIP_CALL( SCIPduplicateMemoryArray(scip, &(*exprhdlr)->name, name, strlen(name)+1) );
   if( desc != NULL )
   {
      SCIP_CALL( SCIPduplicateMemoryArray(scip, &(*exprhdlr)->desc, desc, strlen(desc)+1) );
   }

   (*exprhdlr)->precedence = precedence;
   (*exprhdlr)->eval = eval;
   (*exprhdlr)->data = data;

   /* create clocks */
   SCIP_CALL( SCIPcreateClock(scip, &(*exprhdlr)->sepatime) );
   SCIP_CALL( SCIPcreateClock(scip, &(*exprhdlr)->proptime) );
   SCIP_CALL( SCIPcreateClock(scip, &(*exprhdlr)->intevaltime) );
   SCIP_CALL( SCIPcreateClock(scip, &(*exprhdlr)->simplifytime) );

   ENSUREBLOCKMEMORYARRAYSIZE(scip, conshdlrdata->exprhdlrs, conshdlrdata->exprhdlrssize, conshdlrdata->nexprhdlrs+1);

   conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs] = *exprhdlr;
   ++conshdlrdata->nexprhdlrs;

   return SCIP_OKAY;
}

/** set the expression handler callbacks to copy and free an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrCopyFreeHdlr(
   SCIP*                      scip,              /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,          /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,          /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRCOPYHDLR((*copyhdlr)), /**< handler copy callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRFREEHDLR((*freehdlr))  /**< handler free callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->copyhdlr = copyhdlr;
   exprhdlr->freehdlr = freehdlr;

   return SCIP_OKAY;
}

/** set the expression handler callbacks to copy and free expression data */
SCIP_RETCODE SCIPsetConsExprExprHdlrCopyFreeData(
   SCIP*                      scip,              /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,          /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,          /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRCOPYDATA((*copydata)), /**< expression data copy callback (can be NULL for expressions without data) */
   SCIP_DECL_CONSEXPR_EXPRFREEDATA((*freedata))  /**< expression data free callback (can be NULL if data does not need to be freed) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->copydata = copydata;
   exprhdlr->freedata = freedata;

   return SCIP_OKAY;
}

/** set the print callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrPrint(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRPRINT((*print))    /**< print callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->print = print;

   return SCIP_OKAY;
}

/** set the parse callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrParse(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRPARSE((*parse))    /**< parse callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->parse = parse;

   return SCIP_OKAY;
}

/** set the curvature detection callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrCurvature(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRCURVATURE((*curvature)) /**< curvature detection callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->curvature = curvature;

   return SCIP_OKAY;
}

/** set the monotonicity detection callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrMonotonicity(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRMONOTONICITY((*monotonicity)) /**< monotonicity detection callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->monotonicity = monotonicity;

   return SCIP_OKAY;
}

/** set the integrality detection callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrIntegrality(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRINTEGRALITY((*integrality)) /**< integrality detection callback (can be NULL) */
   )
{ /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->integrality = integrality;

   return SCIP_OKAY;
}

/** set the hash callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrHash(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRHASH((*hash))      /**< hash callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->hash = hash;

   return SCIP_OKAY;
}

/** set the compare callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrCompare(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRCOMPARE((*compare))/**< compare callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->compare = compare;

   return SCIP_OKAY;
}

/** set the derivative evaluation callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrBwdiff(
            SCIP*                      scip,          /**< SCIP data structure */
            SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
            SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
            SCIP_DECL_CONSEXPR_EXPRBWDIFF((*bwdiff))  /**< derivative evaluation callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->bwdiff = bwdiff;

   return SCIP_OKAY;
}

/** set the interval evaluation callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrIntEval(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRINTEVAL((*inteval))/**< interval evaluation callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->inteval = inteval;

   return SCIP_OKAY;
}

/** set the simplify callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrSimplify(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRSIMPLIFY((*simplify))  /**< simplify callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->simplify = simplify;

   return SCIP_OKAY;
}

/** set the reverse propagation callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrReverseProp(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRREVERSEPROP((*reverseprop))/**< reverse propagation callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->reverseprop = reverseprop;

   return SCIP_OKAY;
}

/** set the separation and estimation callbacks of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrSepa(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRINITSEPA((*initsepa)), /**< separation initialization callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPREXITSEPA((*exitsepa)), /**< separation deinitialization callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRSEPA((*sepa)),     /**< separation callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRESTIMATE((*estimate))  /**< estimator callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->initsepa = initsepa;
   exprhdlr->exitsepa = exitsepa;
   exprhdlr->sepa = sepa;
   exprhdlr->estimate = estimate;

   return SCIP_OKAY;
}

/** set the branching score callback of an expression handler */
SCIP_RETCODE SCIPsetConsExprExprHdlrBranchscore(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr,      /**< expression handler */
   SCIP_DECL_CONSEXPR_EXPRBRANCHSCORE((*brscore)) /**< branching score callback (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(exprhdlr != NULL);

   exprhdlr->brscore = brscore;

   return SCIP_OKAY;
}

/** gives expression handlers */
SCIP_CONSEXPR_EXPRHDLR** SCIPgetConsExprExprHdlrs(
   SCIP_CONSHDLR*             conshdlr       /**< expression constraint handler */
   )
{
   assert(conshdlr != NULL);

   return SCIPconshdlrGetData(conshdlr)->exprhdlrs;
}

/** gives number of expression handlers */
int SCIPgetConsExprExprNHdlrs(
   SCIP_CONSHDLR*             conshdlr       /**< expression constraint handler */
   )
{
   assert(conshdlr != NULL);

   return SCIPconshdlrGetData(conshdlr)->nexprhdlrs;
}

/** returns an expression handler of a given name (or NULL if not found) */
SCIP_CONSEXPR_EXPRHDLR* SCIPfindConsExprExprHdlr(
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   const char*                name           /**< name of expression handler */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   int h;

   assert(conshdlr != NULL);
   assert(name != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   for( h = 0; h < conshdlrdata->nexprhdlrs; ++h )
      if( strcmp(SCIPgetConsExprExprHdlrName(conshdlrdata->exprhdlrs[h]), name) == 0 )
         return conshdlrdata->exprhdlrs[h];

   return NULL;
}

/** returns expression handler for variable expressions */
SCIP_CONSEXPR_EXPRHDLR* SCIPgetConsExprExprHdlrVar(
   SCIP_CONSHDLR*             conshdlr       /**< expression constraint handler */
   )
{
   assert(conshdlr != NULL);

   return SCIPconshdlrGetData(conshdlr)->exprvarhdlr;
}

/** returns expression handler for constant value expressions */
SCIP_CONSEXPR_EXPRHDLR* SCIPgetConsExprExprHdlrValue(
   SCIP_CONSHDLR*             conshdlr       /**< expression constraint handler */
   )
{
   assert(conshdlr != NULL);

   return SCIPconshdlrGetData(conshdlr)->exprvalhdlr;
}

/** returns expression handler for sum expressions */
SCIP_CONSEXPR_EXPRHDLR* SCIPgetConsExprExprHdlrSum(
   SCIP_CONSHDLR*             conshdlr       /**< expression constraint handler */
   )
{
   assert(conshdlr != NULL);

   return SCIPconshdlrGetData(conshdlr)->exprsumhdlr;
}

/** returns expression handler for product expressions */
SCIP_CONSEXPR_EXPRHDLR* SCIPgetConsExprExprHdlrProduct(
   SCIP_CONSHDLR*             conshdlr       /**< expression constraint handler */
   )
{
   assert(conshdlr != NULL);

   return SCIPconshdlrGetData(conshdlr)->exprprodhdlr;
}

/** gives the name of an expression handler */
const char* SCIPgetConsExprExprHdlrName(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
)
{
   assert(exprhdlr != NULL);

   return exprhdlr->name;
}

/** gives the description of an expression handler (can be NULL) */
const char* SCIPgetConsExprExprHdlrDescription(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
)
{
   assert(exprhdlr != NULL);

   return exprhdlr->desc;
}

/** gives the precedence of an expression handler */
unsigned int SCIPgetConsExprExprHdlrPrecedence(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
)
{
   assert(exprhdlr != NULL);

   return exprhdlr->precedence;
}

/** gives the data of an expression handler */
SCIP_CONSEXPR_EXPRHDLRDATA* SCIPgetConsExprExprHdlrData(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr      /**< expression handler */
)
{
   assert(exprhdlr != NULL);

   return exprhdlr->data;
}
/** returns whether expression handler implements the print callback */
SCIP_Bool SCIPhasConsExprExprHdlrPrint(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
   )
{
   assert(exprhdlr != NULL);

   return exprhdlr->print != NULL;
}

/** returns whether expression handler implements the interval evaluation callback */
SCIP_Bool SCIPhasConsExprExprHdlrIntEval(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
   )
{
   assert(exprhdlr != NULL);

   return exprhdlr->inteval != NULL;
}

/** returns whether expression handler implements the estimator callback */
SCIP_Bool SCIPhasConsExprExprHdlrEstimate(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
   )
{
   assert(exprhdlr != NULL);

   return exprhdlr->estimate != NULL;
}

/** returns whether expression handler implements the simplification callback */
SCIP_Bool SCIPhasConsExprExprHdlrSimplify(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
   )
{
   assert(exprhdlr != NULL);

   return exprhdlr->simplify != NULL;
}

/** returns whether expression handler implements the reverse propagation callback */
SCIP_Bool SCIPhasConsExprExprHdlrReverseProp(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
   )
{
   assert(exprhdlr != NULL);

   return exprhdlr->reverseprop != NULL;
}

/** returns whether expression handler implements the initialization callback */
SCIP_Bool SCIPhasConsExprExprHdlrInitSepa(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
   )
{
   assert(exprhdlr != NULL);

   return exprhdlr->initsepa != NULL;
}

/** returns whether expression handler implements the deinitialization callback */
SCIP_Bool SCIPhasConsExprExprHdlrExitSepa(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
   )
{
   assert(exprhdlr != NULL);

   return exprhdlr->exitsepa != NULL;
}

/** returns whether expression handler implements the separation callback */
SCIP_Bool SCIPhasConsExprExprHdlrSepa(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
   )
{
   assert(exprhdlr != NULL);

   return exprhdlr->sepa != NULL;
}

/** returns whether expression handler implements the branching score callback */
SCIP_Bool SCIPhasConsExprExprHdlrBranchingScore(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr       /**< expression handler */
   )
{
   assert(exprhdlr != NULL);

   return exprhdlr->brscore != NULL;
}

/** calls the print callback of an expression handler */
SCIP_DECL_CONSEXPR_EXPRPRINT(SCIPprintConsExprExprHdlr)
{
   assert(scip != NULL);
   assert(expr != NULL);

   if( SCIPhasConsExprExprHdlrPrint(expr->exprhdlr) )
   {
      SCIP_CALL( (*expr->exprhdlr->print)(scip, expr, stage, currentchild, parentprecedence, file) );
   }
   else
   {
      /* default: <hdlrname>(<child1>, <child2>, ...) */
      switch( stage )
      {
         case SCIP_CONSEXPRITERATOR_ENTEREXPR :
         {
            SCIPinfoMessage(scip, file, SCIPgetConsExprExprHdlrName(expr->exprhdlr));
            if( SCIPgetConsExprExprNChildren(expr) > 0 )
            {
               SCIPinfoMessage(scip, file, "(");
            }
            break;
         }

         case SCIP_CONSEXPRITERATOR_VISITEDCHILD :
         {
            if( currentchild < SCIPgetConsExprExprNChildren(expr)-1 )
            {
               SCIPinfoMessage(scip, file, ", ");
            }
            else
            {
               SCIPinfoMessage(scip, file, ")");
            }

            break;
         }

         case SCIP_CONSEXPRITERATOR_VISITINGCHILD :
         case SCIP_CONSEXPRITERATOR_LEAVEEXPR :
         default:
            break;
      }
   }

   return SCIP_OKAY;
}

/** calls the parse callback of an expression handler */
SCIP_DECL_CONSEXPR_EXPRPARSE(SCIPparseConsExprExprHdlr)
{
   assert(scip != NULL);
   assert(exprhdlr != NULL);
   assert(expr != NULL);
   assert(success != NULL);

   *expr = NULL;

   if( exprhdlr->parse == NULL )
   {
      /* TODO we could just look for a comma separated list of operands and try to initialize the expr with this one?
       * That would be sufficient for sin, cos, exp, log, abs, for example.
       */
      SCIPdebugMessage("Expression handler <%s> has no parsing method.\n", SCIPgetConsExprExprHdlrName(exprhdlr));
      *success = FALSE;
      return SCIP_OKAY;
   }

   /* give control to exprhdlr's parser */
   SCIP_CALL( exprhdlr->parse(scip, consexprhdlr, exprhdlr, string, endstring, expr, success) );

   assert(*success || (*expr == NULL));

   return SCIP_OKAY;
}

/** calls the expression hash callback */
SCIP_DECL_CONSEXPR_EXPRHASH(SCIPhashConsExprExprHdlr)
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(hashkey != NULL);

   if( expr->exprhdlr->hash != NULL )
   {
      SCIP_CALL( (*expr->exprhdlr->hash)(scip, expr, hashkey, childrenhashes) );
   }
   else
   {
      int i;

      /* compute initial hash from expression handler name if callback is not implemented
       * this can lead to more collisions and thus a larger number of expensive expression compare calls
       */
      *hashkey = 0;
      for( i = 0; expr->exprhdlr->name[i] != '\0'; i++ )
         *hashkey += (unsigned int) expr->exprhdlr->name[i]; /*lint !e571*/

      *hashkey = SCIPcalcFibHash((SCIP_Real)*hashkey);

      /* now make use of the hashkeys of the children */
      for( i = 0; i < expr->nchildren; ++i )
         *hashkey ^= childrenhashes[i];
   }

   return SCIP_OKAY;
}

/** calls the expression compare callback */
SCIP_DECL_CONSEXPR_EXPRCOMPARE(SCIPcompareConsExprExprHdlr)
{
   assert(expr1 != NULL);
   assert(expr2 != NULL);
   assert(expr1->exprhdlr == expr2->exprhdlr);

   if( expr1->exprhdlr->compare != NULL )
   {
      /* enforces OR1-OR4 */
      return expr1->exprhdlr->compare(expr1, expr2);
   }
   else
   {
      /* enforces OR5: default comparison method of expressions of the same type:
       * expr1 < expr2 if and only if expr1_i = expr2_i for all i < k and expr1_k < expr2_k.
       * if there is no such k, use number of children to decide
       * if number of children is equal, both expressions are equal
       * @note: Warning, this method doesn't know about expression data. So if your expressions have special data,
       * you must implement the compare callback: SCIP_DECL_CONSEXPR_EXPRCMP
       */
      int i;
      int nchildren1;
      int nchildren2;
      int compareresult;

      nchildren1 = SCIPgetConsExprExprNChildren(expr1);
      nchildren2 = SCIPgetConsExprExprNChildren(expr2);

      for( i = 0; i < nchildren1 && i < nchildren2; ++i )
      {
         compareresult = SCIPcompareConsExprExprs(SCIPgetConsExprExprChildren(expr1)[i], SCIPgetConsExprExprChildren(expr2)[i]);
         if( compareresult != 0 )
            return compareresult;
      }

      return nchildren1 == nchildren2 ? 0 : nchildren1 < nchildren2 ? -1 : 1;
   }
}

/** calls the evaluation callback of an expression handler
 *
 * further, allows to evaluate w.r.t. given children values
 */
SCIP_RETCODE SCIPevalConsExprExprHdlr(
   SCIP*                      scip,         /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*        expr,         /**< expression */
   SCIP_Real*                 val,          /**< buffer store value of expression */
   SCIP_Real*                 childrenvals, /**< values for children, or NULL if values stored in children should be used */
   SCIP_SOL*                  sol           /**< solution that is evaluated (used by the var-expression) */
)
{
   SCIP_Real* origvals = NULL;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(expr->exprhdlr != NULL);
   assert(expr->exprhdlr->eval != NULL);
   assert(val != NULL);

   /* temporarily overwrite the evalvalue in all children with values from childrenvals */
   if( childrenvals != NULL && expr->nchildren > 0 )
   {
      int c;

      SCIP_CALL( SCIPallocBufferArray(scip, &origvals, expr->nchildren) );

      for( c = 0; c < expr->nchildren; ++c )
      {
         origvals[c] = expr->children[c]->evalvalue;
         expr->children[c]->evalvalue = childrenvals[c];
      }
   }

   /* call expression eval callback */
   SCIP_CALL( expr->exprhdlr->eval(scip, expr, val, sol) );

   /* restore original evalvalues in children */
   if( origvals != NULL )
   {
      int c;
      for( c = 0; c < expr->nchildren; ++c )
         expr->children[c]->evalvalue = origvals[c];

      SCIPfreeBufferArray(scip, &origvals);
   }

   return SCIP_OKAY;
}

/** calls the expression interval evaluation callback */
SCIP_DECL_CONSEXPR_EXPRINTEVAL(SCIPintevalConsExprExprHdlr)
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(interval != NULL);

   if( SCIPhasConsExprExprHdlrIntEval(expr->exprhdlr) )
   {
      SCIP_CALL( SCIPstartClock(scip, expr->exprhdlr->intevaltime) );
      SCIP_CALL( expr->exprhdlr->inteval(scip, expr, interval, intevalvar, intevalvardata) );
      SCIP_CALL( SCIPstopClock(scip, expr->exprhdlr->intevaltime) );
   }

   return SCIP_OKAY;
}

/** calls estimator method of expression handler */
SCIP_DECL_CONSEXPR_EXPRESTIMATE(SCIPestimateConsExprExprHdlr)
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(coefs != NULL);
   assert(success != NULL);

   *success = FALSE;

   if( SCIPhasConsExprExprHdlrEstimate(expr->exprhdlr) )
   {
      SCIP_CALL( SCIPstartClock(scip, expr->exprhdlr->sepatime) );
      SCIP_CALL( expr->exprhdlr->estimate(scip, conshdlr, expr, sol, overestimate, targetvalue, coefs, constant, islocal, success) );
      SCIP_CALL( SCIPstopClock(scip, expr->exprhdlr->sepatime) );

      /* update statistics */
      ++expr->exprhdlr->nsepacalls;
   }

   return SCIP_OKAY;
}

/** calls the simplification method of an expression handler */
SCIP_DECL_CONSEXPR_EXPRSIMPLIFY(SCIPsimplifyConsExprExprHdlr)
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(simplifiedexpr != NULL);

   if( SCIPhasConsExprExprHdlrSimplify(expr->exprhdlr) )
   {
      SCIP_CALL( SCIPstartClock(scip, expr->exprhdlr->simplifytime) );
      SCIP_CALL( expr->exprhdlr->simplify(scip, expr, simplifiedexpr) );
      SCIP_CALL( SCIPstopClock(scip, expr->exprhdlr->simplifytime) );

      /* update statistics */
      ++(expr->exprhdlr->nsimplifycalls);
      if( expr != *simplifiedexpr )
         ++(expr->exprhdlr->nsimplified);
   }

   return SCIP_OKAY;
}

/** calls the expression callback for reverse propagation */
SCIP_DECL_CONSEXPR_EXPRREVERSEPROP(SCIPreversepropConsExprExprHdlr)
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(reversepropqueue != NULL);
   assert(infeasible != NULL);
   assert(nreductions != NULL);

   *infeasible = FALSE;
   *nreductions = 0;

   if( SCIPhasConsExprExprHdlrReverseProp(expr->exprhdlr) )
   {
      SCIP_CALL( SCIPstartClock(scip, expr->exprhdlr->proptime) );
      SCIP_CALL( expr->exprhdlr->reverseprop(scip, expr, reversepropqueue, infeasible, nreductions, force) );
      SCIP_CALL( SCIPstopClock(scip, expr->exprhdlr->proptime) );

      /* update statistics */
      expr->exprhdlr->ndomreds += *nreductions;
      if( *infeasible )
         ++(expr->exprhdlr->ncutoffs);
      ++(expr->exprhdlr->npropcalls);
   }

   return SCIP_OKAY;
}

/** calls the separation initialization method of an expression handler */
SCIP_DECL_CONSEXPR_EXPRINITSEPA(SCIPinitsepaConsExprExprHdlr)
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(infeasible != NULL);

   *infeasible = FALSE;

   if( SCIPhasConsExprExprHdlrInitSepa(expr->exprhdlr) )
   {
      SCIP_CALL( SCIPstartClock(scip, expr->exprhdlr->sepatime) );
      SCIP_CALL( expr->exprhdlr->initsepa(scip, conshdlr, expr, overestimate, underestimate, infeasible) );
      SCIP_CALL( SCIPstopClock(scip, expr->exprhdlr->sepatime) );

      /* update statistics */
      if( *infeasible )
         ++(expr->exprhdlr->ncutoffs);
      ++(expr->exprhdlr->nsepacalls);
   }

   return SCIP_OKAY;
}

/** calls the separation deinitialization method of an expression handler */
SCIP_DECL_CONSEXPR_EXPREXITSEPA(SCIPexitsepaConsExprExprHdlr)
{
   assert(scip != NULL);
   assert(expr != NULL);

   if( SCIPhasConsExprExprHdlrExitSepa(expr->exprhdlr) )
   {
      SCIP_CALL( SCIPstartClock(scip, expr->exprhdlr->sepatime) );
      SCIP_CALL( expr->exprhdlr->exitsepa(scip, expr) );
      SCIP_CALL( SCIPstopClock(scip, expr->exprhdlr->sepatime) );
   }

   return SCIP_OKAY;
}

/** calls separator method of expression handler to separate a given solution */
SCIP_DECL_CONSEXPR_EXPRSEPA(SCIPsepaConsExprExprHdlr)
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(result != NULL);
   assert(ncuts != NULL);

   *result = SCIP_DIDNOTRUN;
   *ncuts = 0;

   if( SCIPhasConsExprExprHdlrSepa(expr->exprhdlr) )
   {
      SCIP_CALL( SCIPstartClock(scip, expr->exprhdlr->sepatime) );
      SCIP_CALL( expr->exprhdlr->sepa(scip, conshdlr, expr, sol, overestimate, mincutviolation, result, ncuts) );
      SCIP_CALL( SCIPstopClock(scip, expr->exprhdlr->sepatime) );

      /* update statistics */
      if( *result == SCIP_CUTOFF )
         ++(expr->exprhdlr->ncutoffs);
      expr->exprhdlr->ncutsfound += *ncuts;
      ++(expr->exprhdlr->nsepacalls);
   }

   return SCIP_OKAY;
}

/** calls the expression branching score callback */
SCIP_DECL_CONSEXPR_EXPRBRANCHSCORE(SCIPbranchscoreConsExprExprHdlr)
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(success != NULL);

   *success = FALSE;

   if( SCIPhasConsExprExprHdlrBranchingScore(expr->exprhdlr) )
   {
      SCIP_CALL( expr->exprhdlr->brscore(scip, expr, sol, auxvalue, brscoretag, success) );

      if( *success )
         SCIPincrementConsExprExprHdlrNBranchScore(expr->exprhdlr);
   }

   return SCIP_OKAY;
}

/** increments the branching score count of an expression handler */
void SCIPincrementConsExprExprHdlrNBranchScore(
   SCIP_CONSEXPR_EXPRHDLR*    exprhdlr
   )
{
   assert(exprhdlr != NULL);

   ++exprhdlr->nbranchscores;
}

/** creates and captures an expression with given expression data and children */
SCIP_RETCODE SCIPcreateConsExprExpr(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR**    expr,             /**< pointer where to store expression */
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr,         /**< expression handler */
   SCIP_CONSEXPR_EXPRDATA* exprdata,         /**< expression data (expression assumes ownership) */
   int                     nchildren,        /**< number of children */
   SCIP_CONSEXPR_EXPR**    children          /**< children (can be NULL if nchildren is 0) */
   )
{
   SCIP_CALL( createExpr(scip, expr, exprhdlr, exprdata, nchildren, children) );

   return SCIP_OKAY;
}

/** creates and captures an expression with up to two children */
SCIP_RETCODE SCIPcreateConsExprExpr2(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          consexprhdlr,     /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR**    expr,             /**< pointer where to store expression */
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr,         /**< expression handler */
   SCIP_CONSEXPR_EXPRDATA* exprdata,         /**< expression data */
   SCIP_CONSEXPR_EXPR*     child1,           /**< first child (can be NULL) */
   SCIP_CONSEXPR_EXPR*     child2            /**< second child (can be NULL) */
   )
{  /*lint --e{715}*/
   assert(expr != NULL);
   assert(exprhdlr != NULL);

   if( child1 != NULL && child2 != NULL )
   {
      SCIP_CONSEXPR_EXPR* pair[2];
      pair[0] = child1;
      pair[1] = child2;

      SCIP_CALL( SCIPcreateConsExprExpr(scip, expr, exprhdlr, exprdata, 2, pair) );
   }
   else if( child2 == NULL )
   {
      SCIP_CALL( SCIPcreateConsExprExpr(scip, expr, exprhdlr, exprdata, child1 == NULL ? 0 : 1, &child1) );
   }
   else
   {
      /* child2 != NULL, child1 == NULL */
      SCIP_CALL( SCIPcreateConsExprExpr(scip, expr, exprhdlr, exprdata, 1, &child2) );
   }

   return SCIP_OKAY;
}

/** creates and captures an expression from a node in an (old-style) expression graph */
SCIP_RETCODE SCIPcreateConsExprExpr3(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          consexprhdlr,     /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR**    expr,             /**< pointer where to store expression */
   SCIP_EXPRGRAPH*         exprgraph,        /**< expression graph */
   SCIP_EXPRGRAPHNODE*     node              /**< expression graph node */
   )
{
   SCIP_CONSEXPR_EXPR** children = NULL;
   int nchildren;
   int c = 0;

   assert(expr != NULL);
   assert(node != NULL);

   *expr = NULL;
   nchildren = SCIPexprgraphGetNodeNChildren(node);

   if( nchildren > 0 )
   {
      SCIP_CALL( SCIPallocBufferArray(scip, &children, nchildren) );

      for( c = 0; c < nchildren; ++c )
      {
         SCIP_CALL( SCIPcreateConsExprExpr3(scip, consexprhdlr, &children[c], exprgraph, SCIPexprgraphGetNodeChildren(node)[c]) );
         if( children[c] == NULL )
            goto TERMINATE;
      }

   }

   switch( SCIPexprgraphGetNodeOperator(node) )
   {
      case SCIP_EXPR_CONST :
         SCIP_CALL( SCIPcreateConsExprExprValue(scip, consexprhdlr, expr, SCIPexprgraphGetNodeOperatorReal(node)) );
         break;

      case SCIP_EXPR_VARIDX :
      {
         int varidx;

         varidx = SCIPexprgraphGetNodeOperatorIndex(node);
         assert(varidx >= 0);
         assert(varidx < SCIPexprgraphGetNVars(exprgraph));

         SCIP_CALL( SCIPcreateConsExprExprVar(scip, consexprhdlr, expr, (SCIP_VAR*)SCIPexprgraphGetVars(exprgraph)[varidx]) );

         break;
      }

      case SCIP_EXPR_PLUS:
      {
         assert(nchildren == 2);
         assert(children != NULL && children[0] != NULL && children[1] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprSum(scip, consexprhdlr, expr, 2, children, NULL, 0.0) );

         break;
      }

      case SCIP_EXPR_MINUS:
      {
         SCIP_Real coefs[2] = {1.0, -1.0};

         assert(nchildren == 2);
         assert(children != NULL && children[0] != NULL && children[1] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprSum(scip, consexprhdlr, expr, 2, children, coefs, 0.0) );

         break;
      }

      case SCIP_EXPR_MUL:
      {
         assert(nchildren == 2);
         assert(children != NULL && children[0] != NULL && children[1] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprProduct(scip, consexprhdlr, expr, 2, children, 1.0) );

         break;
      }

      case SCIP_EXPR_DIV:
      {
         SCIP_CONSEXPR_EXPR* factors[2];

         assert(nchildren == 2);
         assert(children != NULL && children[0] != NULL && children[1] != NULL);

         factors[0] = children[0];
         SCIP_CALL( SCIPcreateConsExprExprPow(scip, consexprhdlr, &factors[1], children[1], -1.0) );
         SCIP_CALL( SCIPcreateConsExprExprProduct(scip, consexprhdlr, expr, 2, factors, 1.0) );

         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &factors[1]) );

         break;
      }

      case SCIP_EXPR_SQUARE:
      {
         assert(nchildren == 1);
         assert(children != NULL && children[0] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprPow(scip, consexprhdlr, expr, *children, 2.0) );

         break;
      }

      case SCIP_EXPR_SQRT:
      {
         assert(nchildren == 1);
         assert(children != NULL && children[0] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprPow(scip, consexprhdlr, expr, *children, 0.5) );

         break;
      }

      case SCIP_EXPR_REALPOWER:
      {
         SCIP_Real exponent;

         exponent = SCIPexprgraphGetNodeRealPowerExponent(node);

         assert(nchildren == 1);
         assert(children != NULL && children[0] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprPow(scip, consexprhdlr, expr, *children, exponent) );

         break;
      }

      case SCIP_EXPR_INTPOWER:
      {
         SCIP_Real exponent;

         exponent = (SCIP_Real)SCIPexprgraphGetNodeIntPowerExponent(node);

         assert(nchildren == 1);
         assert(children != NULL && children[0] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprPow(scip, consexprhdlr, expr, *children, exponent) );

         break;
      }

      case SCIP_EXPR_SUM:
      {
         SCIP_CALL( SCIPcreateConsExprExprSum(scip, consexprhdlr, expr, nchildren, children, NULL, 0.0) );

         break;
      }

      case SCIP_EXPR_PRODUCT:
      {
         SCIP_CALL( SCIPcreateConsExprExprProduct(scip, consexprhdlr, expr, nchildren, children, 1.0) );

         break;
      }

      case SCIP_EXPR_LINEAR:
      {
         SCIP_CALL( SCIPcreateConsExprExprSum(scip, consexprhdlr, expr, nchildren, children, SCIPexprgraphGetNodeLinearCoefs(node), SCIPexprgraphGetNodeLinearConstant(node)) );

         break;
      }

      case SCIP_EXPR_QUADRATIC:
      {
         SCIP_QUADELEM quadelem;
         SCIP_CONSEXPR_EXPR* prod;
         int i;

         SCIP_CALL( SCIPcreateConsExprExprSum(scip, consexprhdlr, expr, 0, NULL, NULL, SCIPexprgraphGetNodeQuadraticConstant(node)) );

         /* append linear terms */
         if( SCIPexprgraphGetNodeQuadraticLinearCoefs(node) != NULL )
         {
            for( i = 0; i < nchildren; ++i )
            {
               if( SCIPexprgraphGetNodeQuadraticLinearCoefs(node)[i] != 0.0 )
               {
                  assert(children != NULL);
                  SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, *expr, children[i], SCIPexprgraphGetNodeQuadraticLinearCoefs(node)[i]) );
               }
            }
         }

         /* append quadratic terms */
         for( i = 0; i < SCIPexprgraphGetNodeQuadraticNQuadElements(node); ++i )
         {
            quadelem = SCIPexprgraphGetNodeQuadraticQuadElements(node)[i];

            if( quadelem.idx1 == quadelem.idx2 )
            {
               assert(children != NULL);
               SCIP_CALL( SCIPcreateConsExprExprPow(scip, consexprhdlr, &prod, children[quadelem.idx1], 2.0) );
            }
            else
            {
               SCIP_CONSEXPR_EXPR* prodchildren[2];

               assert(children != NULL);

               prodchildren[0] = children[quadelem.idx1];
               prodchildren[1] = children[quadelem.idx2];

               SCIP_CALL( SCIPcreateConsExprExprProduct(scip, consexprhdlr, &prod, 2, prodchildren, 1.0) );
            }

            SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, *expr, prod, quadelem.coef) );

            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &prod) );
         }

         break;
      }

      case SCIP_EXPR_POLYNOMIAL:
      {
         SCIP_EXPRDATA_MONOMIAL* monom;
         int m;

         SCIP_CALL( SCIPcreateConsExprExprSum(scip, consexprhdlr, expr, 0, NULL, NULL, SCIPexprgraphGetNodePolynomialConstant(node)) );

         /* append monomials */
         for( m = 0; m < SCIPexprgraphGetNodePolynomialNMonomials(node); ++m )
         {
            SCIP_Real* exponents;

            monom = SCIPexprgraphGetNodePolynomialMonomials(node)[m];
            exponents = SCIPexprGetMonomialExponents(monom);

            if( SCIPexprGetMonomialNFactors(monom) == 1 && (exponents == NULL || exponents[0] == 1.0) )
            {
               assert(children != NULL && children[SCIPexprGetMonomialChildIndices(monom)[0]] != NULL);

               /* monom is linear in child -> append child itself */
               SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, *expr, children[SCIPexprGetMonomialChildIndices(monom)[0]], SCIPexprGetMonomialCoef(monom)) );
            }
            else
            {
               /* monom is nonlinear -> translate into a product expression */
               SCIP_CONSEXPR_EXPR* monomial;
               int f;

               SCIP_CALL( SCIPcreateConsExprExprProduct(scip, consexprhdlr, &monomial, 0, NULL, 1.0) );

               for( f = 0; f < SCIPexprGetMonomialNFactors(monom); ++f )
               {
                  assert(children != NULL && children[SCIPexprGetMonomialChildIndices(monom)[f]] != NULL);
                  if( exponents == NULL || exponents[f] == 1.0 )
                  {
                     SCIP_CALL( SCIPappendConsExprExprProductExpr(scip, monomial, children[SCIPexprGetMonomialChildIndices(monom)[f]]) );
                  }
                  else
                  {
                     SCIP_CONSEXPR_EXPR* powexpr;

                     SCIP_CALL( SCIPcreateConsExprExprPow(scip, consexprhdlr, &powexpr, children[SCIPexprGetMonomialChildIndices(monom)[f]], exponents[f]) );
                     SCIP_CALL( SCIPappendConsExprExprProductExpr(scip, monomial, powexpr) );
                     SCIP_CALL( SCIPreleaseConsExprExpr(scip, &powexpr) );
                  }
               }

               SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, *expr, monomial, SCIPexprGetMonomialCoef(monom)) );
               SCIP_CALL( SCIPreleaseConsExprExpr(scip, &monomial) );
            }
         }

         break;
      }

      case SCIP_EXPR_EXP:
      {
         assert(nchildren == 1);
         assert(children != NULL && children[0] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprExp(scip, consexprhdlr, expr, children[0]) );

         break;
      }
      case SCIP_EXPR_LOG:
      {
         assert(nchildren == 1);
         assert(children != NULL && children[0] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprLog(scip, consexprhdlr, expr, children[0]) );

         break;
      }
      case SCIP_EXPR_ABS:
      {
         assert(nchildren == 1);
         assert(children != NULL && children[0] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprAbs(scip, consexprhdlr, expr, children[0]) );

         break;
      }
      case SCIP_EXPR_SIN:
      {
         assert(nchildren == 1);
         assert(children != NULL && children[0] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprSin(scip, consexprhdlr, expr, children[0]) );

         break;
      }
      case SCIP_EXPR_COS:
      {
         assert(nchildren == 1);
         assert(children != NULL && children[0] != NULL);

         SCIP_CALL( SCIPcreateConsExprExprCos(scip, consexprhdlr, expr, children[0]) );

         break;
      }
      case SCIP_EXPR_SIGNPOWER:
      case SCIP_EXPR_TAN:
      case SCIP_EXPR_MIN:
      case SCIP_EXPR_MAX:
      case SCIP_EXPR_SIGN:
      case SCIP_EXPR_USER:
      case SCIP_EXPR_PARAM:
      case SCIP_EXPR_LAST:
      default:
         goto TERMINATE;
   }


TERMINATE:
   /* release all created children expressions (c-1...0) */
   for( --c; c >= 0; --c )
   {
      assert(children != NULL && children[c] != NULL);
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &children[c]) );
   }

   SCIPfreeBufferArrayNull(scip, &children);

   return SCIP_OKAY;
}

/** appends child to the children list of expr */
SCIP_RETCODE SCIPappendConsExprExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression */
   SCIP_CONSEXPR_EXPR*   child               /**< expression to be appended */
   )
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(child != NULL);
   assert(expr->monotonicitysize == 0);  /* should not append child while mononoticity is stored in expr (not updated here) */
   assert(expr->nlocksneg == 0);  /* should not append child while expression is locked (not updated here) */
   assert(expr->nlockspos == 0);  /* should not append child while expression is locked (not updated here) */

   ENSUREBLOCKMEMORYARRAYSIZE(scip, expr->children, expr->childrensize, expr->nchildren + 1);

   expr->children[expr->nchildren] = child;
   ++expr->nchildren;

   /* capture child */
   SCIPcaptureConsExprExpr(child);

   return SCIP_OKAY;
}

/** overwrites/replaces a child of an expressions
 *
 * @note the old child is released and the newchild is captured, unless they are the same (=same pointer)
 */
SCIP_RETCODE SCIPreplaceConsExprExprChild(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression which is going to replace a child */
   int                     childidx,         /**< index of child being replaced */
   SCIP_CONSEXPR_EXPR*     newchild          /**< the new child */
   )
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(newchild != NULL);
   assert(childidx < SCIPgetConsExprExprNChildren(expr));
   assert(expr->monotonicitysize == 0);  /* should not append child while mononoticity is stored in expr (not updated here) */
   assert(expr->nlocksneg == 0);  /* should not append child while expression is locked (not updated here) */
   assert(expr->nlockspos == 0);  /* should not append child while expression is locked (not updated here) */

   /* do nothing if child is not changing */
   if( newchild == expr->children[childidx] )
      return SCIP_OKAY;

   /* capture new child (do this before releasing the old child in case there are equal */
   SCIPcaptureConsExprExpr(newchild);

   SCIP_CALL( SCIPreleaseConsExprExpr(scip, &(expr->children[childidx])) );
   expr->children[childidx] = newchild;

   return SCIP_OKAY;
}

/** duplicates the given expression
 *
 * If a copy could not be created (e.g., due to missing copy callbacks in expression handlers), *copyexpr will be set to NULL.
 */
SCIP_RETCODE SCIPduplicateConsExprExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr,       /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*   expr,               /**< original expression */
   SCIP_CONSEXPR_EXPR**  copyexpr            /**< buffer to store duplicate of expr */
   )
{
   SCIP_CALL( copyExpr(scip, scip, consexprhdlr, expr, copyexpr, NULL, NULL) );

   return SCIP_OKAY;
}

/** gets the number of times the expression is currently captured */
int SCIPgetConsExprExprNUses(
   SCIP_CONSEXPR_EXPR*   expr               /**< expression */
   )
{
   assert(expr != NULL);

   return expr->nuses;
}

/** captures an expression (increments usage count) */
void SCIPcaptureConsExprExpr(
   SCIP_CONSEXPR_EXPR*   expr               /**< expression */
   )
{
   assert(expr != NULL);

   ++expr->nuses;
}

/** releases an expression (decrements usage count and possibly frees expression) */
SCIP_RETCODE SCIPreleaseConsExprExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR**  rootexpr            /**< pointer to expression to be released */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSEXPR_EXPR* expr;

   assert(rootexpr != NULL);
   assert(*rootexpr != NULL);
   assert((*rootexpr)->nuses > 0);

   if( (*rootexpr)->nuses > 1 )
   {
      --(*rootexpr)->nuses;
      *rootexpr = NULL;

      return SCIP_OKAY;
   }

   /* handle the root expr separately: free enfodata and expression data here */
   SCIP_CALL( freeEnfoData(scip, *rootexpr, TRUE) );

   if( (*rootexpr)->exprdata != NULL && (*rootexpr)->exprhdlr->freedata != NULL )
   {
      SCIP_CALL( (*rootexpr)->exprhdlr->freedata(scip, *rootexpr) );
   }

   SCIP_CALL( SCIPexpriteratorCreate(&it, SCIPfindConshdlr(scip, CONSHDLR_NAME), SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, *rootexpr, SCIP_CONSEXPRITERATOR_DFS, TRUE) );
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_VISITINGCHILD | SCIP_CONSEXPRITERATOR_VISITEDCHILD);
   for( expr = SCIPexpriteratorGetCurrent(it); !SCIPexpriteratorIsEnd(it) ; )
   {
      /* expression should be used by its parent and maybe by the iterator (only the root!)
       * in VISITEDCHILD we assert that expression is only used by its parent
       */
      assert(expr != NULL);
      assert(0 <= expr->nuses && expr->nuses <= 2);

      switch( SCIPexpriteratorGetStageDFS(it) )
      {
         case SCIP_CONSEXPRITERATOR_VISITINGCHILD :
         {
            /* check whether a child needs to be visited (nuses == 1)
             * if not, then we still have to release it
             */
            SCIP_CONSEXPR_EXPR* child;

            child = SCIPexpriteratorGetChildExprDFS(it);
            if( child->nuses > 1 )
            {
               /* child is not going to be freed: just release it */
               SCIP_CALL( SCIPreleaseConsExprExpr(scip, &child) );
               expr = SCIPexpriteratorSkipDFS(it);
               continue;
            }

            assert(child->nuses == 1);

            /* free child's enfodata and expression data when entering child */
            SCIP_CALL( freeEnfoData(scip, child, TRUE) );

            if( child->exprdata != NULL )
            {
               if( child->exprhdlr->freedata != NULL )
               {
                  SCIP_CALL( child->exprhdlr->freedata(scip, child) );
                  assert(child->exprdata == NULL);
               }
               else
               {
                  child->exprdata = NULL;
               }
            }

            break;
         }

         case SCIP_CONSEXPRITERATOR_VISITEDCHILD :
         {
            /* free child after visiting it */
            SCIP_CONSEXPR_EXPR* child;

            child = SCIPexpriteratorGetChildExprDFS(it);
            /* child should only be used by its parent */
            assert(child->nuses == 1);

            /* child should have no data associated */
            assert(child->exprdata == NULL);

            /* free child expression */
            SCIP_CALL( freeExpr(scip, &child) );
            expr->children[SCIPexpriteratorGetChildIdxDFS(it)] = NULL;

            break;
         }

         default:
            SCIPABORT(); /* we should never be called in this stage */
            break;
      }

      expr = SCIPexpriteratorGetNext(it);
   }

   SCIPexpriteratorFree(&it);

   /* handle the root expr separately: free its children and itself here */
   SCIP_CALL( freeExpr(scip, rootexpr) );

   return SCIP_OKAY;
}

/** gives the number of children of an expression */
int SCIPgetConsExprExprNChildren(
   SCIP_CONSEXPR_EXPR*   expr               /**< expression */
   )
{
   assert(expr != NULL);

   return expr->nchildren;
}

/** gives the children of an expression (can be NULL if no children) */
SCIP_CONSEXPR_EXPR** SCIPgetConsExprExprChildren(
   SCIP_CONSEXPR_EXPR*   expr               /**< expression */
   )
{
   assert(expr != NULL);

   return expr->children;
}

/** gets the handler of an expression
 *
 * This identifies the type of the expression (sum, variable, ...).
 */
SCIP_CONSEXPR_EXPRHDLR* SCIPgetConsExprExprHdlr(
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   assert(expr != NULL);

   return expr->exprhdlr;
}

/** gets the expression data of an expression */
SCIP_CONSEXPR_EXPRDATA* SCIPgetConsExprExprData(
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   assert(expr != NULL);

   return expr->exprdata;
}

/** returns whether an expression is a variable expression */
SCIP_Bool SCIPisConsExprExprVar(
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   assert(expr != NULL);

   return strcmp(expr->exprhdlr->name, "var") == 0;
}

/** returns the variable used for linearizing a given expression (return value might be NULL)
 *
 * @note for variable expression it returns the corresponding variable
 */
SCIP_VAR* SCIPgetConsExprExprAuxVar(
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   assert(expr != NULL);

   return SCIPisConsExprExprVar(expr) ? SCIPgetConsExprExprVarVar(expr) : expr->auxvar;
}

/** sets the expression data of an expression
 *
 * The pointer to possible old data is overwritten and the
 * freedata-callback is not called before.
 * This function is intended to be used by expression handler.
 */
void SCIPsetConsExprExprData(
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression */
   SCIP_CONSEXPR_EXPRDATA* exprdata          /**< expression data to be set (can be NULL) */
   )
{
   assert(expr != NULL);

   expr->exprdata = exprdata;
}

/** print an expression as info-message */
SCIP_RETCODE SCIPprintConsExprExpr(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          consexprhdlr,     /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression to be printed */
   FILE*                   file              /**< file to print to, or NULL for stdout */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSEXPRITERATOR_STAGE stage;
   int currentchild;
   unsigned int parentprecedence;

   assert(scip != NULL);
   assert(consexprhdlr != NULL);
   assert(expr != NULL);

   SCIP_CALL( SCIPexpriteratorCreate(&it, consexprhdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, expr, SCIP_CONSEXPRITERATOR_DFS, TRUE) );
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_ALLSTAGES);

   while( !SCIPexpriteratorIsEnd(it) )
   {
      assert(expr->exprhdlr != NULL);
      stage = SCIPexpriteratorGetStageDFS(it);

      if( stage == SCIP_CONSEXPRITERATOR_VISITEDCHILD || stage == SCIP_CONSEXPRITERATOR_VISITINGCHILD )
         currentchild = SCIPexpriteratorGetChildIdxDFS(it);
      else
         currentchild = -1;

      if( SCIPexpriteratorGetParentDFS(it) != NULL )
         parentprecedence = SCIPgetConsExprExprHdlrPrecedence(SCIPgetConsExprExprHdlr(SCIPexpriteratorGetParentDFS(it)));
      else
         parentprecedence = 0;

      SCIP_CALL( SCIPprintConsExprExprHdlr(scip, expr, stage, currentchild, parentprecedence, file) );

      expr = SCIPexpriteratorGetNext(it);
   }

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** initializes printing of expressions in dot format */
SCIP_RETCODE SCIPprintConsExprExprDotInit(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          consexprhdlr,     /**< expression constraint handler */
   SCIP_CONSEXPR_PRINTDOTDATA** dotdata,     /**< buffer to store dot printing data */
   FILE*                   file,             /**< file to print to, or NULL for stdout */
   SCIP_CONSEXPR_PRINTDOT_WHAT whattoprint   /**< info on what to print for each expression */
   )
{
   assert(scip != NULL);
   assert(consexprhdlr != NULL);
   assert(dotdata != NULL);

   if( file == NULL )
      file = stdout;

   SCIP_CALL( SCIPallocBlockMemory(scip, dotdata) );

   (*dotdata)->file = file;
   SCIP_CALL( SCIPexpriteratorCreate(&(*dotdata)->iterator, consexprhdlr, SCIPblkmem(scip)) );
   (*dotdata)->closefile = FALSE;
   (*dotdata)->whattoprint = whattoprint;
   SCIP_CALL( SCIPhashmapCreate(&(*dotdata)->leaveexprs, SCIPblkmem(scip), 100) );

   SCIPinfoMessage(scip, file, "strict digraph exprgraph {\n");
   SCIPinfoMessage(scip, file, "node [fontcolor=white, style=filled, rankdir=LR]\n");

   return SCIP_OKAY;
}

/** initializes printing of expressions in dot format to a file with given filename */
SCIP_RETCODE SCIPprintConsExprExprDotInit2(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          consexprhdlr,     /**< expression constraint handler */
   SCIP_CONSEXPR_PRINTDOTDATA** dotdata,     /**< buffer to store dot printing data */
   const char*             filename,         /**< name of file to print to */
   SCIP_CONSEXPR_PRINTDOT_WHAT whattoprint   /**< info on what to print for each expression */
   )
{
   FILE* f;

   assert(dotdata != NULL);
   assert(filename != NULL);

   f = fopen(filename, "w");
   if( f == NULL )
   {
      SCIPerrorMessage("could not open file <%s> for writing\n", filename);  /* error code would be in errno */
      return SCIP_FILECREATEERROR;
   }

   SCIP_CALL( SCIPprintConsExprExprDotInit(scip, consexprhdlr, dotdata, f, whattoprint) );
   (*dotdata)->closefile = TRUE;

   return SCIP_OKAY;
}

SCIP_RETCODE SCIPprintConsExprExprDot(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSEXPR_PRINTDOTDATA* dotdata,      /**< data as initialized by \ref SCIPprintConsExprExprDotInit() */
   SCIP_CONSEXPR_EXPR*     expr              /**< expression to be printed */
   )
{
   SCIP_Real color;
   int c;

   assert(scip != NULL);
   assert(dotdata != NULL);
   assert(expr != NULL);
   assert(expr->exprhdlr != NULL);

   SCIP_CALL( SCIPexpriteratorInit(dotdata->iterator, expr, SCIP_CONSEXPRITERATOR_DFS, FALSE) );

   while( !SCIPexpriteratorIsEnd(dotdata->iterator) )
   {
      /* print expression as dot node */

      if( SCIPgetConsExprExprNChildren(expr) == 0 )
      {
         SCIP_CALL( SCIPhashmapInsert(dotdata->leaveexprs, (void*)expr, NULL) );
      }

      /* make up some color from the expression type (it's name) */
      color = 0.0;
      for( c = 0; expr->exprhdlr->name[c] != '\0'; ++c )
         color += (tolower(expr->exprhdlr->name[c]) - 'a') / 26.0;
      color = SCIPfrac(scip, color);
      SCIPinfoMessage(scip, dotdata->file, "n%p [fillcolor=\"%g,%g,%g\", label=\"", expr, color, color, color);

      if( dotdata->whattoprint & SCIP_CONSEXPR_PRINTDOT_EXPRHDLR )
      {
         SCIPinfoMessage(scip, dotdata->file, "%s\\n", SCIPgetConsExprExprHdlrName(expr->exprhdlr));
      }

      if( dotdata->whattoprint & SCIP_CONSEXPR_PRINTDOT_EXPRSTRING )
      {
         SCIP_CALL( SCIPprintConsExprExprHdlr(scip, expr, SCIP_CONSEXPRITERATOR_ENTEREXPR, -1, 0, dotdata->file) );
         for( c = 0; c < expr->nchildren; ++c )
         {
            SCIP_CALL( SCIPprintConsExprExprHdlr(scip, expr, SCIP_CONSEXPRITERATOR_VISITINGCHILD, c, 0, dotdata->file) );
            SCIPinfoMessage(scip, dotdata->file, "c%d", c);
            SCIP_CALL( SCIPprintConsExprExprHdlr(scip, expr, SCIP_CONSEXPRITERATOR_VISITEDCHILD, c, 0, dotdata->file) );
         }
         SCIP_CALL( SCIPprintConsExprExprHdlr(scip, expr, SCIP_CONSEXPRITERATOR_LEAVEEXPR, -1, 0, dotdata->file) );

         SCIPinfoMessage(scip, dotdata->file, "\\n");
      }

      if( dotdata->whattoprint & SCIP_CONSEXPR_PRINTDOT_NUSES )
      {
         /* print number of uses */
         SCIPinfoMessage(scip, dotdata->file, "%d uses\\n", expr->nuses);
      }

      if( dotdata->whattoprint & SCIP_CONSEXPR_PRINTDOT_NUSES )
      {
         /* print number of locks */
         SCIPinfoMessage(scip, dotdata->file, "%d,%d +,-locks\\n", expr->nlockspos, expr->nlocksneg);
      }

      if( dotdata->whattoprint & SCIP_CONSEXPR_PRINTDOT_EVALVALUE )
      {
         /* print eval value */
         SCIPinfoMessage(scip, dotdata->file, "val=%g", expr->evalvalue);

         if( (dotdata->whattoprint & SCIP_CONSEXPR_PRINTDOT_EVALTAG) == SCIP_CONSEXPR_PRINTDOT_EVALTAG )
         {
            /* print also eval tag */
            SCIPinfoMessage(scip, dotdata->file, " (%u)", expr->evaltag);
         }
         SCIPinfoMessage(scip, dotdata->file, "\\n");
      }

      if( dotdata->whattoprint & SCIP_CONSEXPR_PRINTDOT_INTERVAL )
      {
         /* print interval value */
         SCIPinfoMessage(scip, dotdata->file, "[%g,%g]", expr->interval.inf, expr->interval.sup);

         if( (dotdata->whattoprint & SCIP_CONSEXPR_PRINTDOT_INTERVALTAG) == SCIP_CONSEXPR_PRINTDOT_INTERVALTAG )
         {
            /* print also interval eval tag */
            SCIPinfoMessage(scip, dotdata->file, " (%u)", expr->intevaltag);
         }
         SCIPinfoMessage(scip, dotdata->file, "\\n");
      }

      SCIPinfoMessage(scip, dotdata->file, "\"]\n");  /* end of label and end of node */

      /* add edges from expr to its children */
      for( c = 0; c < expr->nchildren; ++c )
         SCIPinfoMessage(scip, dotdata->file, "n%p -> n%p [label=\"c%d\"]\n", (void*)expr, (void*)expr->children[c], c);

      expr = SCIPexpriteratorGetNext(dotdata->iterator);
   }

   return SCIP_OKAY;
}

/** finishes printing of expressions in dot format */
SCIP_RETCODE SCIPprintConsExprExprDotFinal(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSEXPR_PRINTDOTDATA** dotdata      /**< buffer where dot printing data has been stored */
   )
{
   SCIP_CONSEXPR_EXPR* expr;
   SCIP_HASHMAPENTRY* entry;
   FILE* file;
   int i;

   assert(dotdata != NULL);
   assert(*dotdata != NULL);

   file = (*dotdata)->file;
   assert(file != NULL);

   /* iterate through all entries of the map */
   SCIPinfoMessage(scip, file, "{rank=same;");
   for( i = 0; i < SCIPhashmapGetNEntries((*dotdata)->leaveexprs); ++i )
   {
      entry = SCIPhashmapGetEntry((*dotdata)->leaveexprs, i);

      if( entry != NULL )
      {
         expr = (SCIP_CONSEXPR_EXPR*) SCIPhashmapEntryGetOrigin(entry);
         assert(expr != NULL);
         assert(SCIPgetConsExprExprNChildren(expr) == 0);

         SCIPinfoMessage(scip, file, " n%p", expr);
      }
   }
   SCIPinfoMessage(scip, file, "}\n");

   SCIPinfoMessage(scip, file, "}\n");

   SCIPhashmapFree(&(*dotdata)->leaveexprs);

   SCIPexpriteratorFree(&(*dotdata)->iterator);

   if( (*dotdata)->closefile )
      fclose((*dotdata)->file);

   SCIPfreeBlockMemory(scip, dotdata);

   return SCIP_OKAY;
}

/** shows a single expression by use of dot and gv
 *
 * This function is meant for debugging purposes.
 * It prints the expression into a temporary file in dot format, then calls dot to create a postscript file, then calls ghostview (gv) to show the file.
 * SCIP will hold until ghostscript is closed.
 */
SCIP_RETCODE SCIPshowConsExprExpr(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*     expr              /**< expression to be printed */
   )
{
   /* this function is for developers, so don't bother with C variants that don't have popen() */
#if _POSIX_C_SOURCE < 2
   SCIPerrorMessage("No POSIX version 2. Try http://distrowatch.com/.");
   return SCIP_ERROR;
#else
   SCIP_CONSEXPR_PRINTDOTDATA* dotdata;
   FILE* f;

   assert(expr != NULL);

   /* call dot to generate postscript output and show it via ghostview */
   f = popen("dot -Tps | gv -", "w");
   if( f == NULL )
   {
      SCIPerrorMessage("Calling popen() failed");
      return SCIP_FILECREATEERROR;
   }

   /* print all of the expression into the pipe */
   SCIP_CALL( SCIPprintConsExprExprDotInit(scip, SCIPfindConshdlr(scip, CONSHDLR_NAME), &dotdata, f, SCIP_CONSEXPR_PRINTDOT_ALL) );
   SCIP_CALL( SCIPprintConsExprExprDot(scip, dotdata, expr) );
   SCIP_CALL( SCIPprintConsExprExprDotFinal(scip, &dotdata) );

   /* close the pipe */
   (void) pclose(f);

   return SCIP_OKAY;
#endif
}

/** prints structure of an expression a la Maple's dismantle */
SCIP_RETCODE SCIPdismantleConsExprExpr(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*     expr              /**< expression to dismantle */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;
   int depth = -1;

   SCIP_CALL( SCIPexpriteratorCreate(&it, SCIPfindConshdlr(scip, CONSHDLR_NAME), SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, expr, SCIP_CONSEXPRITERATOR_DFS, TRUE) );
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_ENTEREXPR | SCIP_CONSEXPRITERATOR_VISITINGCHILD | SCIP_CONSEXPRITERATOR_LEAVEEXPR);

   for( ; !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
   {
      switch( SCIPexpriteratorGetStageDFS(it) )
      {
         case SCIP_CONSEXPRITERATOR_ENTEREXPR:
         {
            int nspaces;
            const char* type;

            ++depth;
            nspaces = 3 * depth;
            type = SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr));

            /* use depth of expression to align output */
            SCIPinfoMessage(scip, NULL, "%*s[%s]: ", nspaces, "", type);

            if( strcmp(type, "var") == 0 )
            {
               SCIP_VAR* var;

               var = SCIPgetConsExprExprVarVar(expr);
               SCIPinfoMessage(scip, NULL, "%s in [%g, %g]\n", SCIPvarGetName(var), SCIPvarGetLbLocal(var),
                  SCIPvarGetUbLocal(var));
            }
            else if(strcmp(type, "sum") == 0)
               SCIPinfoMessage(scip, NULL, "%g\n", SCIPgetConsExprExprSumConstant(expr));
            else if(strcmp(type, "prod") == 0)
               SCIPinfoMessage(scip, NULL, "%g\n", SCIPgetConsExprExprProductCoef(expr));
            else if(strcmp(type, "val") == 0)
               SCIPinfoMessage(scip, NULL, "%g\n", SCIPgetConsExprExprValueValue(expr));
            else if(strcmp(type, "pow") == 0)
               SCIPinfoMessage(scip, NULL, "%g\n", SCIPgetConsExprExprPowExponent(expr));
            else if(strcmp(type, "exp") == 0)
               SCIPinfoMessage(scip, NULL, "\n");
            else if(strcmp(type, "log") == 0)
               SCIPinfoMessage(scip, NULL, "\n");
            else if(strcmp(type, "abs") == 0)
               SCIPinfoMessage(scip, NULL, "\n");
            else
               SCIPinfoMessage(scip, NULL, "NOT IMPLEMENTED YET\n");

            break;
         }

         case SCIP_CONSEXPRITERATOR_VISITINGCHILD:
         {
            int nspaces;
            const char* type;

            nspaces = 3 * depth;
            type = SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr));

            if( strcmp(type, "sum") == 0 )
            {
               SCIPinfoMessage(scip, NULL, "%*s   ", nspaces, "");
               SCIPinfoMessage(scip, NULL, "[coef]: %g\n", SCIPgetConsExprExprSumCoefs(expr)[SCIPexpriteratorGetChildIdxDFS(it)]);
            }

            break;
         }

         case SCIP_CONSEXPRITERATOR_LEAVEEXPR:
         {
            --depth;
            break;
         }

         default:
            /* shouldn't be here */
            SCIPABORT();
            break;
      }
   }

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** Creates an expression from a string.
 * We specify the grammar that defines the syntax of an expression. Loosely speaking, a Base will be any "block",
 * a Factor is a Base to a power, a Term is a product of Factors and an Expression is a sum of terms
 * The actual definition:
 * <pre>
 * Expression -> ["+" | "-"] Term { ("+" | "-" | "number *") ] Term }
 * Term       -> Factor { ("*" | "/" ) Factor }
 * Factor     -> Base [ "^" "number" | "^(" "number" ")" ]
 * Base       -> "number" | "<varname>" | "(" Expression ")" | Op "(" OpExpression ")
 * </pre>
 * where [a|b] means a or b or none, (a|b) means a or b, {a} means 0 or more a.
 *
 * Note that Op and OpExpression are undefined. Op corresponds to the name of an expression handler and
 * OpExpression to whatever string the expression handler accepts (through its parse method).
 *
 * See also @ref parseExpr.
 */
SCIP_RETCODE SCIPparseConsExprExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr,       /**< expression constraint handler */
   const char*           exprstr,            /**< string with the expr to parse */
   const char**          finalpos,           /**< buffer to store the position of exprstr where we finished reading, or NULL if not of interest */
   SCIP_CONSEXPR_EXPR**  expr                /**< pointer to store the expr parsed */
   )
{
   const char* finalpos_;
   SCIP_RETCODE retcode;
   SCIP_HASHMAP* vartoexprvarmap;

   SCIP_CALL( SCIPhashmapCreate(&vartoexprvarmap, SCIPblkmem(scip), 5 * SCIPgetNVars(scip)) );

   /* if parseExpr fails, we still want to free hashmap */
   retcode = parseExpr(scip, consexprhdlr, vartoexprvarmap, exprstr, &finalpos_, expr);

   SCIPhashmapFree(&vartoexprvarmap);

   if( finalpos != NULL )
      *finalpos = finalpos_;

   return retcode;
}

/** evaluate an expression in a point
 *
 * Iterates over expressions to also evaluate children, if necessary.
 * Value can be received via SCIPgetConsExprExprEvalValue().
 * If an evaluation error (division by zero, ...) occurs, this value will
 * be set to SCIP_INVALID.
 *
 * If a nonzero \p soltag is passed, then only (sub)expressions are
 * reevaluated that have a different solution tag. If a soltag of 0
 * is passed, then subexpressions are always reevaluated.
 * The tag is stored together with the value and can be received via
 * SCIPgetConsExprExprEvalTag().
 */
SCIP_RETCODE SCIPevalConsExprExpr(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          consexprhdlr,     /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression to be evaluated */
   SCIP_SOL*               sol,              /**< solution to be evaluated */
   unsigned int            soltag            /**< tag that uniquely identifies the solution (with its values), or 0. */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;

   assert(scip != NULL);
   assert(consexprhdlr != NULL);
   assert(expr != NULL);

   /* if value is up-to-date, then nothing to do */
   if( soltag != 0 && expr->evaltag == soltag )
      return SCIP_OKAY;

   /* assume we'll get a domain error, so we don't have to get this expr back if we abort the iteration
    * if there is no domain error, then we will overwrite the evalvalue in the last leaveexpr stage
    */
   expr->evalvalue = SCIP_INVALID;
   expr->evaltag = soltag;

   SCIP_CALL( SCIPexpriteratorCreate(&it, consexprhdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, expr, SCIP_CONSEXPRITERATOR_DFS, TRUE) );
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_VISITINGCHILD | SCIP_CONSEXPRITERATOR_LEAVEEXPR);

   while( !SCIPexpriteratorIsEnd(it) )
   {
      switch( SCIPexpriteratorGetStageDFS(it) )
      {
         case SCIP_CONSEXPRITERATOR_VISITINGCHILD :
         {
            SCIP_CONSEXPR_EXPR* child;

            if( soltag == 0 )
               break;

            /* check whether child has been evaluated for that solution already */
            child = SCIPexpriteratorGetChildExprDFS(it);
            if( soltag == child->evaltag )
            {
               if( child->evalvalue == SCIP_INVALID ) /*lint !e777*/
                  goto TERMINATE;

               /* skip this child
                * this already returns the next one, so continue with loop
                */
               expr = SCIPexpriteratorSkipDFS(it);
               continue;
            }

            break;
         }

         case SCIP_CONSEXPRITERATOR_LEAVEEXPR :
         {
            SCIP_CALL( SCIPevalConsExprExprHdlr(scip, expr, &expr->evalvalue, NULL, sol) );
            expr->evaltag = soltag;

            if( expr->evalvalue == SCIP_INVALID ) /*lint !e777*/
               goto TERMINATE;

            break;
         }

         default :
            /* we should never be here */
            SCIPABORT();
            break;
      }

      expr = SCIPexpriteratorGetNext(it);
   }

TERMINATE:
   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** gives the value from the last evaluation of an expression (or SCIP_INVALID if there was an eval error) */
SCIP_Real SCIPgetConsExprExprValue(
   SCIP_CONSEXPR_EXPR*     expr              /**< expression */
   )
{
   assert(expr != NULL);

   return expr->evalvalue;
}

/** sets the evaluation value */
void SCIPsetConsExprExprEvalValue(
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression */
   SCIP_Real               value,            /**< value to set */
   unsigned int            tag               /**< tag of solution that was evaluated, or 0 */
   )
{
   assert(expr != NULL);

   expr->evalvalue = value;
   expr->evaltag = tag;
}

/** gives the evaluation tag from the last evaluation, or 0 */
unsigned int SCIPgetConsExprExprEvalTag(
   SCIP_CONSEXPR_EXPR*     expr              /**< expression */
   )
{
   assert(expr != NULL);

   return expr->evaltag;
}

/** computes the gradient for a given point
 *
 * Initiates an expression walk to also evaluate children, if necessary.
 * Value can be received via SCIPgetConsExprExprPartialDiff().
 * If an error (division by zero, ...) occurs, this value will
 * be set to SCIP_INVALID.
 */
SCIP_RETCODE SCIPcomputeConsExprExprGradient(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          consexprhdlr,     /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*     rootexpr,         /**< expression to be evaluated */
   SCIP_SOL*               sol,              /**< solution to be evaluated (NULL for the current LP solution) */
   unsigned int            soltag            /**< tag that uniquely identifies the solution (with its values), or 0. */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSEXPR_EXPR* expr;
   SCIP_CONSEXPR_EXPR* child;
   SCIP_Real derivative;
   unsigned int difftag;

   assert(scip != NULL);
   assert(consexprhdlr != NULL);
   assert(rootexpr != NULL);

   /* ensure expression is evaluated */
   SCIP_CALL( SCIPevalConsExprExpr(scip, consexprhdlr, rootexpr, sol, soltag) );

   /* check if expression could not be evaluated */
   if( SCIPgetConsExprExprValue(rootexpr) == SCIP_INVALID ) /*lint !e777*/
   {
      rootexpr->derivative = SCIP_INVALID;
      return SCIP_OKAY;
   }

   conshdlrdata = SCIPconshdlrGetData(consexprhdlr);
   assert(conshdlrdata != NULL);

   if( rootexpr->exprhdlr == conshdlrdata->exprvalhdlr )
   {
      rootexpr->derivative = 0.0;
      return SCIP_OKAY;
   }

   difftag = ++(conshdlrdata->lastdifftag);

   rootexpr->derivative = 1.0;
   rootexpr->difftag = difftag;

   SCIP_CALL( SCIPexpriteratorCreate(&it, consexprhdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, rootexpr, SCIP_CONSEXPRITERATOR_DFS, TRUE) );
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_VISITINGCHILD);

   for( expr = SCIPexpriteratorGetCurrent(it); !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
   {
      assert(expr->evalvalue != SCIP_INVALID); /*lint !e777*/

      if( expr->exprhdlr->bwdiff == NULL )
      {
         rootexpr->derivative = SCIP_INVALID;
         break;
      }

      child = SCIPexpriteratorGetChildExprDFS(it);
      assert(child != NULL);

      /* reset the value of the partial derivative w.r.t. a variable expression if we see it for the first time */
      if( child->difftag != difftag && SCIPisConsExprExprVar(child) )
         child->derivative = 0.0;

      /* update differentiation tag of the child */
      child->difftag = difftag;

      /* call backward differentiation callback */
      if( child->exprhdlr == conshdlrdata->exprvalhdlr )
      {
         derivative = 0.0;
      }
      else
      {
         derivative = SCIP_INVALID;
         SCIP_CALL( (*expr->exprhdlr->bwdiff)(scip, expr, SCIPexpriteratorGetChildIdxDFS(it), &derivative) );

         if( derivative == SCIP_INVALID ) /*lint !e777*/
         {
            rootexpr->derivative = SCIP_INVALID;
            break;
         }
      }

      /* update partial derivative stored in the child expression
       * for a variable, we have to sum up the partial derivatives of the root w.r.t. this variable over all parents
       * for other intermediate expressions, we only store the partial derivative of the root w.r.t. this expression
       */
      if( !SCIPisConsExprExprVar(child) )
         child->derivative = expr->derivative * derivative;
      else
         child->derivative += expr->derivative * derivative;
   }

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** returns the partial derivative of an expression w.r.t. a variable (or SCIP_INVALID if there was an evaluation error) */
SCIP_Real SCIPgetConsExprExprPartialDiff(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr,       /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression which has been used in the last SCIPcomputeConsExprExprGradient() call */
   SCIP_VAR*             var                 /**< variable (needs to be in the expression) */
   )
{
   SCIP_CONSEXPR_EXPR* varexpr;
   SCIP_HASHMAP* var2expr;

   assert(scip != NULL);
   assert(consexprhdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(consexprhdlr), CONSHDLR_NAME) == 0);
   assert(expr != NULL);
   assert(var != NULL);
   assert(expr->exprhdlr != SCIPgetConsExprExprHdlrValue(consexprhdlr) || expr->derivative == 0.0);

   /* check if an error occurred during the last SCIPcomputeConsExprExprGradient() call */
   if( strcmp(expr->exprhdlr->name, "val") == 0 )
      return 0.0;

   /* return 0.0 for value expression */
   if( expr->derivative == SCIP_INVALID ) /*lint !e777*/
      return SCIP_INVALID;

   /* use variable to expressions mapping which is stored as the expression handler data */
   var2expr = (SCIP_HASHMAP*)SCIPgetConsExprExprHdlrData(SCIPgetConsExprExprHdlrVar(consexprhdlr));
   assert(var2expr != NULL);
   assert(SCIPhashmapExists(var2expr, var));

   varexpr = (SCIP_CONSEXPR_EXPR*)SCIPhashmapGetImage(var2expr, var);
   assert(varexpr != NULL);
   assert(SCIPisConsExprExprVar(varexpr));

   /* use difftag to decide whether the variable belongs to the expression */
   return (expr->difftag != varexpr->difftag) ? 0.0 : varexpr->derivative;
}

/** returns the derivative stored in an expression (or SCIP_INVALID if there was an evaluation error) */
SCIP_Real SCIPgetConsExprExprDerivative(
   SCIP_CONSEXPR_EXPR*     expr              /**< expression */
   )
{
   assert(expr != NULL);

   return expr->derivative;
}

/** evaluates an expression over a box
 *
 * Initiates an expression walk to also evaluate children, if necessary.
 * The resulting interval can be received via SCIPgetConsExprExprEvalInterval().
 * If the box does not overlap with the domain of the function behind the expression
 * (e.g., sqrt([-2,-1]) or 1/[0,0]) this interval will be empty.
 *
 * For variables, the local variable bounds, possibly relaxed by some amount, are used as interval.
 * The actual interval is determined by the intevalvar function, if not NULL.
 * If NULL, then the local bounds of the variable are taken without modification.
 *
 * If a nonzero \p boxtag is passed, then only (sub)expressions are
 * reevaluated that have a different tag. If a tag of 0 is passed,
 * then subexpressions are always reevaluated.
 * The tag is stored together with the interval and can be received via
 * SCIPgetConsExprExprEvalIntervalTag().
 */
SCIP_RETCODE SCIPevalConsExprExprInterval(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          consexprhdlr,     /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression to be evaluated */
   unsigned int            boxtag,           /**< tag that uniquely identifies the current variable domains (with its values), or 0 */
   SCIP_DECL_CONSEXPR_INTEVALVAR((*intevalvar)), /**< function to call to evaluate interval of variable */
   void*                   intevalvardata    /**< data to be passed to intevalvar call */
   )
{
   assert(expr != NULL);

   SCIP_CALL( forwardPropExpr(scip, consexprhdlr, expr, FALSE, FALSE, intevalvar, intevalvardata, boxtag, NULL, NULL) );

   return SCIP_OKAY;
}

/** returns the interval from the last interval evaluation of an expression (interval can be empty) */
SCIP_INTERVAL SCIPgetConsExprExprInterval(
   SCIP_CONSEXPR_EXPR*     expr              /**< expression */
   )
{
   assert(expr != NULL);

   return expr->interval;
}

/** sets the evaluation interval */
void SCIPsetConsExprExprEvalInterval(
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression */
   SCIP_INTERVAL*          interval,         /**< interval to set */
   unsigned int            tag               /**< tag of variable domains that were evaluated, or 0. */
   )
{
   assert(expr != NULL);

   SCIPintervalSetBounds(&expr->interval, SCIPintervalGetInf(*interval), SCIPintervalGetSup(*interval));
   expr->intevaltag = tag;
}

/** gives the box tag from the last interval evaluation, or 0 */
unsigned int SCIPgetConsExprExprEvalIntervalTag(
   SCIP_CONSEXPR_EXPR*     expr              /**< expression */
   )
{
   assert(expr != NULL);

   return expr->intevaltag;
}

/** tightens the bounds of an expression and stores the result in the expression interval; variables in variable
 *  expression will be tightened immediately if SCIP is in a stage above SCIP_STAGE_TRANSFORMED
 *
 *  If a reversepropqueue is given, then the expression will be added to the queue if its bounds could be tightened without detecting infeasibility.
 */
SCIP_RETCODE SCIPtightenConsExprExprInterval(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression to be tightened */
   SCIP_INTERVAL           newbounds,        /**< new bounds for the expression */
   SCIP_Bool               force,            /**< force tightening even if below bound strengthening tolerance */
   SCIP_QUEUE*             reversepropqueue, /**< reverse propagation queue, or NULL if not in reverse propagation */
   SCIP_Bool*              cutoff,           /**< buffer to store whether a node's bounds were propagated to an empty interval */
   int*                    ntightenings      /**< buffer to add the total number of tightenings, or NULL */
   )
{
   SCIP_Real oldlb;
   SCIP_Real oldub;
   SCIP_Real newlb;
   SCIP_Real newub;
   SCIP_Bool tightenlb;
   SCIP_Bool tightenub;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(cutoff != NULL);
   assert(!SCIPintervalIsEmpty(SCIP_INTERVAL_INFINITY, expr->interval));

   oldlb = SCIPintervalGetInf(expr->interval);
   oldub = SCIPintervalGetSup(expr->interval);
   *cutoff = FALSE;

#if 0 /* def SCIP_DEBUG */
   SCIPdebugMsg(scip, "Trying to tighten bounds of expr ");
   SCIP_CALL( SCIPprintConsExprExpr(scip, expr, NULL) );
   SCIPdebugMsgPrint(scip, " from [%g,%g] to [%g,%g]\n", oldlb, oldub, SCIPintervalGetInf(newbounds), SCIPintervalGetSup(newbounds));
#endif

   /* check if the new bounds lead to an empty interval */
   if( SCIPintervalIsEmpty(SCIP_INTERVAL_INFINITY, newbounds) || SCIPintervalGetInf(newbounds) > oldub
      || SCIPintervalGetSup(newbounds) < oldlb )
   {
      SCIPdebugMsg(scip, "cut off due to empty intersection of new bounds [%g,%g] with old bounds [%g,%g]\n", newbounds.inf, newbounds.sup, oldlb, oldub);

      SCIPintervalSetEmpty(&expr->interval);
      *cutoff = TRUE;

      return SCIP_OKAY;
   }

   /* intersect old interval with the new one */
   SCIPintervalIntersect(&expr->interval, expr->interval, newbounds);
   newlb = SCIPintervalGetInf(expr->interval);
   newub = SCIPintervalGetSup(expr->interval);

   /* mark the current problem to be infeasible if either the lower/upper bound is above/below +/- SCIPinfinity() */
   if( SCIPisInfinity(scip, newlb) || SCIPisInfinity(scip, -newub) )
   {
      SCIPdebugMsg(scip, "cut off due to infinite new bounds [%g,%g]\n", newlb, newub);

      SCIPintervalSetEmpty(&expr->interval);
      *cutoff = TRUE;

      return SCIP_OKAY;
   }


   /* check which bound can be tightened */
   if( force )
   {
      tightenlb = !SCIPisHugeValue(scip, -newlb) && SCIPisGT(scip, newlb, oldlb);
      tightenub = !SCIPisHugeValue(scip, newub) && SCIPisLT(scip, newub, oldub);
   }
   else
   {
      tightenlb = !SCIPisHugeValue(scip, -newlb) && SCIPisLbBetter(scip, newlb, oldlb, oldub);
      tightenub = !SCIPisHugeValue(scip, newub) && SCIPisUbBetter(scip, newub, oldlb, oldub);
   }

   /* tighten interval of the expression and variable bounds of linearization variables */
   if( tightenlb || tightenub )
   {
      SCIP_VAR* var;

      /* mark expression as tightened; important for reverse propagation to ignore irrelevant sub-expressions */
      expr->hastightened = TRUE;

      /* tighten bounds of linearization variable
       * but: do not tighten variable in problem stage (important for unittests)
       * TODO put some kind of #ifdef UNITTEST around this once the unittest are modified to include the .c file (again)?
       */
      var = SCIPgetConsExprExprAuxVar(expr);
      if( var != NULL && (SCIPgetStage(scip) == SCIP_STAGE_SOLVING || SCIPgetStage(scip) == SCIP_STAGE_PRESOLVING) )
      {
         SCIP_Bool tightened;

         if( tightenlb )
         {
            SCIP_CALL( SCIPtightenVarLb(scip, var, newlb, force, cutoff, &tightened) );

            if( tightened )
            {
               if( ntightenings != NULL )
                  ++*ntightenings;
               SCIPdebugMsg(scip, "tightened lb on auxvar <%s> to %g\n", SCIPvarGetName(var), newlb);
            }

            if( *cutoff )
               return SCIP_OKAY;
         }

         if( tightenub )
         {
            SCIP_CALL( SCIPtightenVarUb(scip, var, newub, force, cutoff, &tightened) );

            if( tightened )
            {
               if( ntightenings != NULL )
                  ++*ntightenings;
               SCIPdebugMsg(scip, "tightened ub on auxvar <%s> to %g\n", SCIPvarGetName(var), newub);
            }

            if( *cutoff )
               return SCIP_OKAY;
         }
      }

      /* if a reversepropagation queue is given, then add expression to that queue if it has at least one child and could have a reverseprop callback */
      if( reversepropqueue != NULL && !expr->inqueue && (expr->nenfos > 0 || SCIPhasConsExprExprHdlrReverseProp(expr->exprhdlr)) )
      {
         /* @todo put children which are in the queue to the end of it! */
         SCIP_CALL( SCIPqueueInsert(reversepropqueue, (void*) expr) );
         expr->inqueue = TRUE;
      }
   }

   return SCIP_OKAY;
}

/** adds branching score to an expression
 *
 * Adds a score to the expression-specific branching score.
 * The branchscoretag argument is used to identify whether the score in the expression needs to be reset before adding a new score.
 * In an expression with children, the scores are distributed to its children.
 * In an expression that is a variable, the score may be used to identify a variable for branching.
 */
void SCIPaddConsExprExprBranchScore(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression where to add branching score */
   unsigned int            branchscoretag,   /**< tag to identify current branching scores */
   SCIP_Real               branchscore       /**< branching score to add to expression */
   )
{
   assert(scip != NULL);
   assert(expr != NULL);
   assert(branchscore >= 0.0);

   /* reset branching score if the tag has changed */
   if( expr->brscoretag != branchscoretag )
   {
      expr->brscore = 0.0;
      expr->brscoretag = branchscoretag;
   }

   expr->brscore += branchscore;
}

/** returns the hash value of an expression */
SCIP_RETCODE SCIPgetConsExprExprHash(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression */
   unsigned int*           hashval           /**< pointer to store the hash value */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(hashval != NULL);

   SCIP_CALL( SCIPexpriteratorCreate(&it, SCIPfindConshdlr(scip, CONSHDLR_NAME), SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, expr, SCIP_CONSEXPRITERATOR_DFS, FALSE) );
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_LEAVEEXPR);

   SCIP_CALL( hashExpr(scip, expr, it, NULL) );

   *hashval = SCIPexpriteratorGetExprUserData(it, expr).uintval;

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}


/** creates and gives the auxiliary variable for a given expression
 *
 * @note if auxiliary variable already present for that expression, then only returns this variable
 * @note for a variable expression it returns the corresponding variable
 */
SCIP_RETCODE SCIPcreateConsExprExprAuxVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression */
   SCIP_VAR**            auxvar              /**< buffer to store pointer to auxiliary variable, or NULL */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_VARTYPE vartype;
   char name[SCIP_MAXSTRLEN];

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(expr != NULL);

   /* if we already have auxvar, then just return it */
   if( expr->auxvar != NULL )
   {
      if( auxvar != NULL )
         *auxvar = expr->auxvar;
      return SCIP_OKAY;
   }

   /* if expression is a variable-expression, then return that variable */
   if( expr->exprhdlr == SCIPgetConsExprExprHdlrVar(conshdlr) )
   {
      if( auxvar != NULL )
         *auxvar = SCIPgetConsExprExprVarVar(expr);
      return SCIP_OKAY;
   }

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->auxvarid >= 0);

   if( expr->exprhdlr == SCIPgetConsExprExprHdlrValue(conshdlr) )
   {
      /* it doesn't harm much to have an auxvar for a constant, but it doesn't seem to make much sense
       * @todo ensure that this will not happen and change the warning to an assert
       */
      SCIPwarningMessage(scip, "Creating auxiliary variable for constant expression.");
   }

   /* create and capture auxiliary variable */
   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "auxvar_%s_%d", expr->exprhdlr->name, conshdlrdata->auxvarid);
   ++conshdlrdata->auxvarid;

   /* type of auxiliary variable depends on integrality information of the expression */
   vartype = SCIPisConsExprExprIntegral(expr) ? SCIP_VARTYPE_IMPLINT : SCIP_VARTYPE_CONTINUOUS;

   SCIP_CALL( SCIPcreateVarBasic(scip, &expr->auxvar, name, MAX( -SCIPinfinity(scip), expr->interval.inf ),
      MIN( SCIPinfinity(scip), expr->interval.sup ), 0.0, vartype) ); /*lint !e666*/
   SCIP_CALL( SCIPaddVar(scip, expr->auxvar) );

   /* mark the auxiliary variable to be invalid after a restart happened; this prevents SCIP to create linear
    * constraints from cuts that contain auxiliary variables
    */
   SCIPvarSetCutInvalidAfterRestart(expr->auxvar, TRUE);

   SCIPdebugMsg(scip, "added auxiliary variable %s [%g,%g] for expression %p\n", SCIPvarGetName(expr->auxvar), SCIPvarGetLbGlobal(expr->auxvar), SCIPvarGetUbGlobal(expr->auxvar), (void*)expr);

   /* add variable locks in both directions */
   SCIP_CALL( SCIPaddVarLocks(scip, expr->auxvar, 1, 1) );

#ifdef WITH_DEBUG_SOLUTION
   if( SCIPdebugIsMainscip(scip) )
   {
      /* store debug solution value of auxiliary variable
       * assumes that expression has been evaluated in debug solution before
       */
      SCIP_CALL( SCIPdebugAddSolVal(scip, expr->auxvar, SCIPgetConsExprExprValue(expr)) );
   }
#endif

   if( auxvar != NULL )
      *auxvar = expr->auxvar;

   return SCIP_OKAY;
}

/** @name Simplifying methods
 *
 * This is largely inspired in Joel Cohen's
 * Computer algebra and symbolic computation: Mathematical methods
 * In particular Chapter 3
 * The other fountain of inspiration is the current simplifying methods in expr.c.
 *
 * Definition of simplified expressions
 * ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 * An expression is simplified if it
 * - is a value expression
 * - is a var expression
 * - is a product expression such that
 *    SP1:  every child is simplified
 *    SP2:  no child is a product
 *    SP4:  no two children are the same expression (those should be multiplied)
 *    SP5:  the children are sorted [commutative rule]
 *    SP7:  no child is a value
 *    SP8:  its coefficient is 1.0 (otherwise should be written as sum)
 *    SP10: it has at least two children
 *    ? at most one child is an exp
 *    ? at most one child is an abs
 *    SP11: no two children are expr*log(expr)
 *    (TODO: we could handle more complicated stuff like x*y*log(x) -> - y * entropy(x), but I am not sure this should
 *    happen at the simplifcation level, or (x*y) * log(x*y), which currently simplifies to x * y * log(x*y))
 * - is a power expression such that
 *    POW1: exponent is not 0
 *    POW2: exponent is not 1
 *    POW3: its child is not a value
 *    POW4: its child is simplified
 *    POW5: if exponent is integer, its child is not a product
 *    POW6: if exponent is integer, its child is not a sum with a single term ((2*x)^2 -> 4*x^2)
 *    POW7: if exponent is 2, its child is not a sum (expand sums)
 *    POW8: if exponent is integer, its child is not a power
 *    POW9: its child is not a sum with a single term with a positive coefficient: (25*x)^0.5 -> 5 x^0.5
 *    POW10: its child is not a binary variable: b^e and e > 0 --> b, b^e and e < 0 --> fix b to 1
 * - is a sum expression such that
 *    SS1: every child is simplified
 *    SS2: no child is a sum
 *    SS3: no child is a value (values should go in the constant of the sum)
 *    SS4: no two children are the same expression (those should be summed up)
 *    SS5: the children are sorted [commutative rule]
 *    SS6: it has at least one child
 *    SS7: if it consists of a single child, then either constant is != 0.0 or coef != 1
 *    SS8: no child has coefficient 0
 *    x if it consists of a single child, then its constant != 0.0 (otherwise, should be written as a product)
 * - it is a function with simplified arguments, but not all of them can be values
 * ? a logarithm doesn't have a product as a child
 * ? the exponent of an exponential is always 1
 *
 * ORDERING RULES
 * ^^^^^^^^^^^^^^
 * These rules define a total order on *simplified* expressions.
 * There are two groups of rules, when comparing equal type expressions and different type expressions
 * Equal type expressions:
 * OR1: u,v value expressions: u < v <=> val(u) < val(v)
 * OR2: u,v var expressions: u < v <=> SCIPvarGetIndex(var(u)) < SCIPvarGetIndex(var(v))
 * OR3: u,v are both sum or product expression: < is a lexicographical order on the terms
 * OR4: u,v are both pow: u < v <=> base(u) < base(v) or, base(u) == base(v) and expo(u) < expo(v)
 * OR5: u,v are u = FUN(u_1, ..., u_n), v = FUN(v_1, ..., v_m): u < v <=> For the first k such that u_k != v_k, u_k < v_k,
 *      or if such a k doesn't exist, then n < m.
 *
 * Different type expressions:
 * OR6: u value, v other: u < v always
 * OR7: u sum, v var or func: u < v <=> u < 0+v
 *      In other words, u = \sum_{i = 1}^n \alpha_i u_i, then u < v <=> u_n < v or if u_n = v and \alpha_n < 1
 * OR8: u product, v pow, sum, var or func: u < v <=> u < 1*v
 *      In other words, u = \Pi_{i = 1}^n u_i,  then u < v <=> u_n < v
 *      @note: since this applies only to simplified expressions, the form of the product is correct. Simplified products
 *             do *not* have constant coefficients
 * OR9: u pow, v sum, var or func: u < v <=> u < v^1
 * OR10: u var, v func: u < v always
 * OR11: u func, v other type of func: u < v <=> name(type(u)) < name(type(v))
 * OR12: none of the rules apply: u < v <=> ! v < u
 * Examples:
 * OR12: x < x^2 ?:  x is var and x^2 product, so none applies.
 *       Hence, we try to answer x^2 < x ?: x^2 < x <=> x < x or if x = x and 2 < 1 <=> 2 < 1 <=> False, so x < x^2 is True
 *       x < x^-1 --OR12--> ~(x^-1 < x) --OR9--> ~(x^-1 < x^1) --OR4--> ~(x < x or -1 < 1) --> ~True --> False
 *       x*y < x --OR8--> x*y < 1*x --OR3--> y < x --OR2--> False
 *       x*y < y --OR8--> x*y < 1*y --OR3--> y < x --OR2--> False
 *
 * Algorithm
 * ^^^^^^^^^
 * The recursive version of the algorithm is
 *
 * EXPR simplify(expr)
 *    for c in 1..expr->nchildren
 *       expr->children[c] = simplify(expr->children[c])
 *    end
 *    return expr->exprhdlr->simplify(expr)
 * end
 *
 * Important: Whatever is returned by a simplify callback **has** to be simplified.
 * Also, all children of the given expression **are** already simplified
 *
 * Here is an outline of the algorithm for simplifying sum expressions:
 * The idea is to create a list of all the children that the simplified expr must containt.
 * We use a linked list to construct it
 *
 * INPUT:  expr  :: sum expression to be simplified
 * OUTPUT: sexpr :: simplified expression
 * NOTE: it *can* modify expr
 *
 * simplified_coefficient <- expr->coefficient
 * expr_list <- empty list (list containing the simplified children of the final simplified expr)
 * For each child in expr->children:
 *    1. if child's coef is 0: continue
 *    2. if child is value: add it to simplified_coefficient and continue
 *    3. if child is not a sum: build list L = [(coef,child)]
 *    4. if child is sum:
 *       4.1. if coef is not 1.0: multiply child by coef (*)
 *       4.2. build list with the children of child, L = [(val, expr) for val in child->coeffs, expr in child->children)]
 *    5. mergeSum(L, expr_list)
 * if expr_list is empty, return value expression with value simplified_coefficient
 * if expr_list has only one child and simplified_coefficient is 1, return child
 * otherwise, build sum expression using the exprs in expr_list as children
 *
 * The method mergeSum simply inserts the elements of L into expr_list. Note that both lists are sorted.
 * While inserting, collisions can happen. A collision means that we have to add the two expressions.
 * However, after adding them, we need to simplify the resulting expression (e.g., the coefficient may become 0.0).
 * Fortunately, the coefficient being 0 is the only case we have to handle.
 * PROOF: all expressions in expr_list are simplified wrt to the sum, meaning that if we would build a sum
 * expression from them, it would yield a simplified sum expression. If there is a collision, then the expression
 * in L has to be in expr_list. The sum yields coef*expr and from the previous one can easily verify that it is
 * a valid child of a simplified sum (it is not a sum, etc), except for the case coef = 0.
 * Note: the context where the proof works is while merging (adding) children. Before this step, the children
 * go through a "local" simplification (i.e., 1-4 above). There, we *do* have to take care of other cases.
 * But, in contrast to products, after this steps, no child in finalchildren is a sum and the proof works.
 *
 * The algorithm for simplifying a product is basically the same. One extra difficulty occurs at (*):
 * The distribution of the exponent over a product children can only happen if the exponent is integral.
 * Also, in that case, the resulting new child could be unsimplified, so it must be re-simplified.
 * While merging, multiplying similar product expressions can render them unsimplified. So to handle it
 * one basically needs to simulate (the new) (*) while merging. Hence, a further merge might be necessary
 * (and then all the book-keeping information to perform the original merge faster is lost)
 *
 * @{
 */

/** compare expressions
 * @return -1, 0 or 1 if expr1 <, =, > expr2, respectively
 * @note: The given expressions are assumed to be simplified.
 */
int SCIPcompareConsExprExprs(
   SCIP_CONSEXPR_EXPR*   expr1,              /**< first expression */
   SCIP_CONSEXPR_EXPR*   expr2               /**< second expression */
   )
{
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr1;
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr2;
   int retval;

   exprhdlr1 = SCIPgetConsExprExprHdlr(expr1);
   exprhdlr2 = SCIPgetConsExprExprHdlr(expr2);

   /* expressions are of the same kind/type; use compare callback or default method */
   if( exprhdlr1 == exprhdlr2 )
   {
      return SCIPcompareConsExprExprHdlr(expr1, expr2);
   }

   /* expressions are of different kind/type */
   /* enforces OR6 */
   if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr1), "val") == 0 )
   {
      return -1;
   }
   /* enforces OR12 */
   if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr2), "val") == 0 )
      return -SCIPcompareConsExprExprs(expr2, expr1);

   /* enforces OR7 */
   if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr1), "sum") == 0 )
   {
      int compareresult;
      int nchildren;

      nchildren = SCIPgetConsExprExprNChildren(expr1);
      compareresult = SCIPcompareConsExprExprs(SCIPgetConsExprExprChildren(expr1)[nchildren-1], expr2);

      if( compareresult != 0 )
         return compareresult;

      /* "base" of the largest expression of the sum is equal to expr2, coefficient might tell us that expr2 is larger */
      if( SCIPgetConsExprExprSumCoefs(expr1)[nchildren-1] < 1.0 )
         return -1;

      /* largest expression of sum is larger or equal than expr2 => expr1 > expr2 */
      return 1;
   }
   /* enforces OR12 */
   if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr2), "sum") == 0 )
      return -SCIPcompareConsExprExprs(expr2, expr1);

   /* enforces OR8 */
   if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr1), "prod") == 0 )
   {
      int compareresult;
      int nchildren;

      nchildren = SCIPgetConsExprExprNChildren(expr1);
      compareresult = SCIPcompareConsExprExprs(SCIPgetConsExprExprChildren(expr1)[nchildren-1], expr2);

      if( compareresult != 0 )
         return compareresult;

      /* largest expression of product is larger or equal than expr2 => expr1 > expr2 */
      return 1;
   }
   /* enforces OR12 */
   if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr2), "prod") == 0 )
      return -SCIPcompareConsExprExprs(expr2, expr1);

   /* enforces OR9 */
   if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr1), "pow") == 0 )
   {
      int compareresult;

      compareresult = SCIPcompareConsExprExprs(SCIPgetConsExprExprChildren(expr1)[0], expr2);

      if( compareresult != 0 )
         return compareresult;

      /* base equal to expr2, exponent might tell us that expr2 is larger */
      if( SCIPgetConsExprExprPowExponent(expr1) < 1.0 )
         return -1;

      /* power expression is larger => expr1 > expr2 */
      return 1;
   }
   /* enforces OR12 */
   if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr2), "pow") == 0 )
      return -SCIPcompareConsExprExprs(expr2, expr1);

   /* enforces OR10 */
   if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr1), "var") == 0 )
      return -1;
   /* enforces OR12 */
   if( strcmp(SCIPgetConsExprExprHdlrName(exprhdlr2), "var") == 0 )
      return -SCIPcompareConsExprExprs(expr2, expr1);

   /* enforces OR11 */
   retval = strcmp(SCIPgetConsExprExprHdlrName(exprhdlr1), SCIPgetConsExprExprHdlrName(exprhdlr2));
   return retval == 0 ? 0 : retval < 0 ? -1 : 1;
}

/** simplifies an expression
 *
 * The given expression will be released and overwritten with the simplified expression.
 * To keep the expression, duplicate it via SCIPduplicateConsExprExpr before calling this method.
 */
SCIP_RETCODE SCIPsimplifyConsExprExpr(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          conshdlr,         /**< constraint handler */
   SCIP_CONSEXPR_EXPR*     rootexpr,         /**< expression to be simplified */
   SCIP_CONSEXPR_EXPR**    simplified,       /**< buffer to store simplified expression */
   SCIP_Bool*              changed           /**< buffer to store if rootexpr actually changed */
)
{
   assert(rootexpr != NULL);
   assert(simplified != NULL);

   SCIP_CALL( reformulateConsExprExpr(scip, conshdlr, rootexpr, TRUE, simplified, changed) );

   return SCIP_OKAY;
}

/**@} */  /* end of simplifying methods */

/** reformulate an expression; this functions works similar as SCIPsimplifyConsExprExpr() but instead of calling the
 *  simplify callback of an expression handler it iterates through all nonlinear handlers and uses the reformulation
 *  callback
 */
SCIP_RETCODE SCIPreformulateConsExprExpr(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          conshdlr,         /**< constraint handler */
   SCIP_CONSEXPR_EXPR*     rootexpr,         /**< expression to be simplified */
   SCIP_CONSEXPR_EXPR**    refrootexpr,      /**< buffer to store reformulated expression */
   SCIP_Bool*              changed           /**< buffer to store if rootexpr actually changed */
   )
{
   assert(rootexpr != NULL);
   assert(refrootexpr != NULL);

   SCIP_CALL( reformulateConsExprExpr(scip, conshdlr, rootexpr, FALSE, refrootexpr, changed) );

   return SCIP_OKAY;
}

/** sets the curvature of an expression */
void SCIPsetConsExprExprCurvature(
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression */
   SCIP_EXPRCURV         curvature           /**< curvature of the expression */
   )
{
   assert(expr != NULL);
   expr->curvature = curvature;
}

/** returns the curvature of an expression */
SCIP_EXPRCURV SCIPgetConsExprExprCurvature(
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   assert(expr != NULL);
   return expr->curvature;
}

/** computes the curvature of a given expression and all its subexpressions
 *
 *  @note this function also evaluates all subexpressions w.r.t. current variable bounds
 */
SCIP_RETCODE SCIPcomputeConsExprExprCurvature(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_CONSHDLR* conshdlr;
   SCIP_EXPRCURV curv;

   assert(scip != NULL);
   assert(expr != NULL);

   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   /* evaluate all subexpressions (not relaxing variable bounds, as not in boundtightening) */
   SCIP_CALL( SCIPevalConsExprExprInterval(scip, conshdlr, expr, 0, NULL, NULL) );

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, expr, SCIP_CONSEXPRITERATOR_DFS, FALSE) );
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_LEAVEEXPR);

   for( expr = SCIPexpriteratorGetCurrent(it); !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) )  /*lint !e441*/
   {
      curv = SCIP_EXPRCURV_UNKNOWN;

      if( expr->exprhdlr->curvature != NULL )
      {
         /* get curvature from expression handler */
         SCIP_CALL( (*expr->exprhdlr->curvature)(scip, conshdlr, expr, &curv) );
      }

      /* set curvature in expression */
      SCIPsetConsExprExprCurvature(expr, curv);
   }

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** returns the monotonicity of an expression w.r.t. to a given child
 *
 *  @note Call SCIPevalConsExprExprInterval before using this function.
 */
SCIP_MONOTONE SCIPgetConsExprExprMonotonicity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression */
   int                   childidx            /**< index of child */
   )
{
   SCIP_MONOTONE monotonicity = SCIP_MONOTONE_UNKNOWN;

   assert(expr != NULL);
   assert(childidx >= 0 || expr->nchildren == 0);
   assert(childidx < expr->nchildren);

   /* check whether the expression handler implements the monotonicity callback */
   if( expr->exprhdlr->monotonicity != NULL )
   {
      SCIP_CALL_ABORT( (*expr->exprhdlr->monotonicity)(scip, expr, childidx, &monotonicity) );
   }

   return monotonicity;
}

/** returns the number of positive rounding locks of an expression */
int SCIPgetConsExprExprNLocksPos(
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   assert(expr != NULL);
   return expr->nlockspos;
}

/** returns the number of negative rounding locks of an expression */
int SCIPgetConsExprExprNLocksNeg(
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   assert(expr != NULL);
   return expr->nlocksneg;
}

/** computes integrality information of a given expression and all its subexpressions; the integrality information can
 * be accessed via SCIPisConsExprExprIntegral()
 */
SCIP_RETCODE SCIPcomputeConsExprExprIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(expr != NULL);

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, expr, SCIP_CONSEXPRITERATOR_DFS, FALSE) );
   SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_LEAVEEXPR);

   for( expr = SCIPexpriteratorGetCurrent(it); !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
   {
      /* compute integrality information */
      expr->isintegral = FALSE;

      if( expr->exprhdlr->integrality != NULL )
      {
         /* get curvature from expression handler */
         SCIP_CALL( (*expr->exprhdlr->integrality)(scip, expr, &expr->isintegral) );
      }
   }

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** returns whether an expression is integral */
SCIP_Bool SCIPisConsExprExprIntegral(
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   assert(expr != NULL);
   return expr->isintegral;
}

/** returns the total number of variables in an expression
 *
 * The function counts variables in common sub-expressions only once.
 */
SCIP_RETCODE SCIPgetConsExprExprNVars(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          conshdlr,         /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression */
   int*                    nvars             /**< buffer to store the total number of variables */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(nvars != NULL);

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, expr, SCIP_CONSEXPRITERATOR_DFS, FALSE) );

   *nvars = 0;
   for( ; !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
      if( SCIPisConsExprExprVar(expr) )
         ++(*nvars);

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/** returns all variable expressions contained in a given expression; the array to store all variable expressions needs
 * to be at least of size the number of unique variables in the expression which is given by SCIpgetConsExprExprNVars()
 * and can be bounded by SCIPgetNVars().
 *
 * @note function captures variable expressions
 */
SCIP_RETCODE SCIPgetConsExprExprVarExprs(
   SCIP*                   scip,             /**< SCIP data structure */
   SCIP_CONSHDLR*          conshdlr,         /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*     expr,             /**< expression */
   SCIP_CONSEXPR_EXPR**    varexprs,         /**< array to store all variable expressions */
   int*                    nvarexprs         /**< buffer to store the total number of variable expressions */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;

   assert(expr != NULL);
   assert(varexprs != NULL);
   assert(nvarexprs != NULL);

   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );
   SCIP_CALL( SCIPexpriteratorInit(it, expr, SCIP_CONSEXPRITERATOR_DFS, FALSE) );

   *nvarexprs = 0;
   for( ; !SCIPexpriteratorIsEnd(it); expr = SCIPexpriteratorGetNext(it) ) /*lint !e441*/
   {
      assert(expr != NULL);

      if( SCIPisConsExprExprVar(expr) )
      {
         /* add variable expression to array and capture expr */
         assert(SCIPgetNTotalVars(scip) >= *nvarexprs + 1);

         varexprs[(*nvarexprs)++] = expr;

         /* capture expression */
         SCIPcaptureConsExprExpr(expr);
      }
   }

   /* @todo sort variable expressions here? */

   SCIPexpriteratorFree(&it);

   return SCIP_OKAY;
}

/*
 * constraint specific interface methods
 */

/** gets the index an expression iterator can use to store iterator specific data in an expression */
SCIP_RETCODE SCIPactivateConsExprExprHdlrIterator(
   SCIP_CONSHDLR*             consexprhdlr,   /**< expression constraint handler */
   int*                       iterindex       /**< buffer to store iteration index */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(consexprhdlr != NULL);
   assert(iterindex != NULL);

   conshdlrdata = SCIPconshdlrGetData(consexprhdlr);
   assert(conshdlrdata != NULL);

   if( conshdlrdata->nactiveiter + 1 >= SCIP_CONSEXPRITERATOR_MAXNACTIVE )
   {
      SCIPerrorMessage("Maximal number of active expression iterators reached.\n");
      return SCIP_MAXDEPTHLEVEL;
   }

   *iterindex = conshdlrdata->nactiveiter++;

   return SCIP_OKAY;
}

/** returns the index that an expression iterator used to store iterator specific data in an expression */
void SCIPdeactivateConsExprExprHdlrIterator(
   SCIP_CONSHDLR*             consexprhdlr,   /**< expression constraint handler */
   int                        iterindex       /**< iteration index that is not used anymore */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(consexprhdlr != NULL);
   assert(iterindex >= 0);

   conshdlrdata = SCIPconshdlrGetData(consexprhdlr);
   assert(conshdlrdata != NULL);

   /* the iterindex must be the one of the last initialized iterator */
   assert(iterindex == conshdlrdata->nactiveiter-1);

   --conshdlrdata->nactiveiter;
}

/** get a new tag that can be used to mark an expression as visited */
unsigned int SCIPgetConsExprExprHdlrNewVisitedTag(
   SCIP_CONSHDLR*             consexprhdlr    /**< expression constraint handler */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(consexprhdlr != NULL);

   conshdlrdata = SCIPconshdlrGetData(consexprhdlr);
   assert(conshdlrdata != NULL);

   return ++conshdlrdata->lastvisitedtag;
}

/** create and include conshdlr to SCIP and set everything except for expression handlers */
static
SCIP_RETCODE includeConshdlrExprBasic(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   /* create expr constraint handler data */
   SCIP_CALL( SCIPallocClearMemory(scip, &conshdlrdata) );
   conshdlrdata->lastsoltag = 1;
   SCIP_CALL( SCIPcreateClock(scip, &conshdlrdata->canonicalizetime) );

   /* include constraint handler */
   SCIP_CALL( SCIPincludeConshdlr(scip, CONSHDLR_NAME, CONSHDLR_DESC,
         CONSHDLR_SEPAPRIORITY, CONSHDLR_ENFOPRIORITY, CONSHDLR_CHECKPRIORITY,
         CONSHDLR_SEPAFREQ, CONSHDLR_PROPFREQ, CONSHDLR_EAGERFREQ, CONSHDLR_MAXPREROUNDS,
         CONSHDLR_DELAYSEPA, CONSHDLR_DELAYPROP, CONSHDLR_NEEDSCONS,
         CONSHDLR_PROP_TIMING, CONSHDLR_PRESOLTIMING,
         conshdlrCopyExpr,
         consFreeExpr, consInitExpr, consExitExpr,
         consInitpreExpr, consExitpreExpr, consInitsolExpr, consExitsolExpr,
         consDeleteExpr, consTransExpr, consInitlpExpr,
         consSepalpExpr, consSepasolExpr, consEnfolpExpr, consEnforelaxExpr, consEnfopsExpr, consCheckExpr,
         consPropExpr, consPresolExpr, consRespropExpr, consLockExpr,
         consActiveExpr, consDeactiveExpr,
         consEnableExpr, consDisableExpr, consDelvarsExpr,
         consPrintExpr, consCopyExpr, consParseExpr,
         consGetVarsExpr, consGetNVarsExpr, consGetDiveBdChgsExpr, conshdlrdata) );

   if( SCIPfindConshdlr(scip, "quadratic") != NULL )
   {
      /* include function that upgrades quadratic constraint to expr constraints */
      SCIP_CALL( SCIPincludeQuadconsUpgrade(scip, quadconsUpgdExpr, QUADCONSUPGD_PRIORITY, TRUE, CONSHDLR_NAME) );
   }

   if( SCIPfindConshdlr(scip, "nonlinear") != NULL )
   {
      /* include the linear constraint upgrade in the linear constraint handler */
      SCIP_CALL( SCIPincludeNonlinconsUpgrade(scip, nonlinconsUpgdExpr, NULL, NONLINCONSUPGD_PRIORITY, TRUE, CONSHDLR_NAME) );
   }

   /* add expr constraint handler parameters */
   SCIP_CALL( SCIPaddIntParam(scip, "constraints/" CONSHDLR_NAME "/maxproprounds",
         "limit on number of propagation rounds for a set of constraints within one round of SCIP propagation",
         &conshdlrdata->maxproprounds, FALSE, 10, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddCharParam(scip, "constraints/" CONSHDLR_NAME "/varboundrelax",
         "strategy on how to relax variable bounds during bound tightening: relax (n)ot, relax by (a)bsolute value, relax by (r)relative value",
         &conshdlrdata->varboundrelax, TRUE, 'a', "nar", NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "constraints/" CONSHDLR_NAME "/varboundrelaxamount",
         "by how much to relax variable bounds during bound tightening if strategy 'a' or 'r'",
         &conshdlrdata->varboundrelaxamount, TRUE, SCIPepsilon(scip), 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "constraints/" CONSHDLR_NAME "/conssiderelaxamount",
         "by how much to relax constraint sides during bound tightening",
         &conshdlrdata->conssiderelaxamount, TRUE, SCIPepsilon(scip), 0.0, 1.0, NULL, NULL) );

   /* include handler for bound change events */
   SCIP_CALL( SCIPincludeEventhdlrBasic(scip, &conshdlrdata->eventhdlr, CONSHDLR_NAME "_boundchange",
         "signals a bound change to an expression constraint", processVarEvent, NULL) );
   assert(conshdlrdata->eventhdlr != NULL);

   /* include table for statistics */
   assert(SCIPfindTable(scip, TABLE_NAME_EXPR) == NULL);
   SCIP_CALL( SCIPincludeTable(scip, TABLE_NAME_EXPR, TABLE_DESC_EXPR, TRUE,
         NULL, NULL, NULL, NULL, NULL, NULL, tableOutputExpr,
         NULL, TABLE_POSITION_EXPR, TABLE_EARLIEST_STAGE_EXPR) );

   return SCIP_OKAY;
}

/** creates the handler for expr constraints and includes it in SCIP */
SCIP_RETCODE SCIPincludeConshdlrExpr(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSHDLR* conshdlr;

   SCIP_CALL( includeConshdlrExprBasic(scip) );

   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* include and remember handler for variable expression */
   SCIP_CALL( SCIPincludeConsExprExprHdlrVar(scip, conshdlr) );
   assert(conshdlrdata->nexprhdlrs > 0 && strcmp(conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1]->name, "var") == 0);
   conshdlrdata->exprvarhdlr = conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1];

   /* include and remember handler for constant value expression */
   SCIP_CALL( SCIPincludeConsExprExprHdlrValue(scip, conshdlr) );
   assert(conshdlrdata->nexprhdlrs > 0 && strcmp(conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1]->name, "val") == 0);
   conshdlrdata->exprvalhdlr = conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1];

   /* include and remember handler for sum expression */
   SCIP_CALL( SCIPincludeConsExprExprHdlrSum(scip, conshdlr) );
   assert(conshdlrdata->nexprhdlrs > 0 && strcmp(conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1]->name, "sum") == 0);
   conshdlrdata->exprsumhdlr = conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1];

   /* include and remember handler for product expression */
   SCIP_CALL( SCIPincludeConsExprExprHdlrProduct(scip, conshdlr) );
   assert(conshdlrdata->nexprhdlrs > 0 && strcmp(conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1]->name, "prod") == 0);
   conshdlrdata->exprprodhdlr = conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1];

   /* include handler for exponential expression */
   SCIP_CALL( SCIPincludeConsExprExprHdlrExp(scip, conshdlr) );
   assert(conshdlrdata->nexprhdlrs > 0 && strcmp(conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1]->name, "exp") == 0);

   /* include handler for logarithmic expression */
   SCIP_CALL( SCIPincludeConsExprExprHdlrLog(scip, conshdlr) );
   assert(conshdlrdata->nexprhdlrs > 0 && strcmp(conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1]->name, "log") == 0);

   /* include handler for absolute expression */
   SCIP_CALL( SCIPincludeConsExprExprHdlrAbs(scip, conshdlr) );
   assert(conshdlrdata->nexprhdlrs > 0 && strcmp(conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1]->name, "abs") == 0);

   /* include handler for power expression */
   SCIP_CALL( SCIPincludeConsExprExprHdlrPow(scip, conshdlr) );
   assert(conshdlrdata->nexprhdlrs > 0 && strcmp(conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1]->name, "pow") == 0);
   conshdlrdata->exprpowhdlr = conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1];

   /* include handler for entropy expression */
   SCIP_CALL( SCIPincludeConsExprExprHdlrEntropy(scip, conshdlr) );
   assert(conshdlrdata->nexprhdlrs > 0 && strcmp(conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1]->name, "entropy") == 0);

   /* include handler for sine expression */
   SCIP_CALL( SCIPincludeConsExprExprHdlrSin(scip, conshdlr) );
   assert(conshdlrdata->nexprhdlrs > 0 && strcmp(conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1]->name, "sin") == 0);

   /* include handler for cosine expression */
   SCIP_CALL( SCIPincludeConsExprExprHdlrCos(scip, conshdlr) );
   assert(conshdlrdata->nexprhdlrs > 0 && strcmp(conshdlrdata->exprhdlrs[conshdlrdata->nexprhdlrs-1]->name, "cos") == 0);

   /* include default nonlinear handler */
   SCIP_CALL( SCIPincludeConsExprNlhdlrDefault(scip, conshdlr) );

   /* include nonlinear handler for quadratics */
   SCIP_CALL( SCIPincludeConsExprNlhdlrQuadratic(scip, conshdlr) );

   /* include nonlinear handler for convex expressions */
   SCIP_CALL( SCIPincludeConsExprNlhdlrConvex(scip, conshdlr) );

   /* include nonlinear handler for bilinear expressions */
   SCIP_CALL( SCIPincludeConsExprNlhdlrBilinear(scip, conshdlr) );

   return SCIP_OKAY;
}

/** includes an expression constraint upgrade method into the expression constraint handler */
SCIP_RETCODE SCIPincludeExprconsUpgrade(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DECL_EXPRCONSUPGD((*exprconsupgd)),  /**< method to call for upgrading expression constraint, or NULL */
   int                   priority,           /**< priority of upgrading method */
   SCIP_Bool             active,             /**< should the upgrading method by active by default? */
   const char*           conshdlrname        /**< name of the constraint handler */
   )
{
   SCIP_CONSHDLR*        conshdlr;
   SCIP_CONSHDLRDATA*    conshdlrdata;
   SCIP_EXPRCONSUPGRADE* exprconsupgrade;
   char                  paramname[SCIP_MAXSTRLEN];
   char                  paramdesc[SCIP_MAXSTRLEN];
   int                   i;

   assert(conshdlrname != NULL );

   /* ignore empty upgrade functions */
   if( exprconsupgd == NULL )
      return SCIP_OKAY;

   /* find the expression constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   if( conshdlr == NULL )
   {
      SCIPerrorMessage("nonlinear constraint handler not found\n");
      return SCIP_PLUGINNOTFOUND;
   }

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* check whether upgrade method exists already */
   for( i = conshdlrdata->nexprconsupgrades - 1; i >= 0; --i )
   {
      if( conshdlrdata->exprconsupgrades[i]->exprconsupgd == exprconsupgd )
      {
#ifdef SCIP_DEBUG
         SCIPwarningMessage(scip, "Try to add already known upgrade method %p for constraint handler <%s>.\n", exprconsupgd, conshdlrname); /*lint !e611*/
#endif
         return SCIP_OKAY;
      }
   }

   /* create an expression constraint upgrade data object */
   SCIP_CALL( SCIPallocBlockMemory(scip, &exprconsupgrade) );
   exprconsupgrade->exprconsupgd = exprconsupgd;
   exprconsupgrade->priority   = priority;
   exprconsupgrade->active     = active;

   /* insert expression constraint upgrade method into constraint handler data */
   assert(conshdlrdata->nexprconsupgrades <= conshdlrdata->exprconsupgradessize);
   if( conshdlrdata->nexprconsupgrades+1 > conshdlrdata->exprconsupgradessize )
   {
      int newsize;

      newsize = SCIPcalcMemGrowSize(scip, conshdlrdata->nexprconsupgrades+1);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &conshdlrdata->exprconsupgrades, conshdlrdata->nexprconsupgrades, newsize) );
      conshdlrdata->exprconsupgradessize = newsize;
   }
   assert(conshdlrdata->nexprconsupgrades+1 <= conshdlrdata->exprconsupgradessize);

   for( i = conshdlrdata->nexprconsupgrades; i > 0 && conshdlrdata->exprconsupgrades[i-1]->priority < exprconsupgrade->priority; --i )
      conshdlrdata->exprconsupgrades[i] = conshdlrdata->exprconsupgrades[i-1];
   assert(0 <= i && i <= conshdlrdata->nexprconsupgrades);
   conshdlrdata->exprconsupgrades[i] = exprconsupgrade;
   conshdlrdata->nexprconsupgrades++;

   /* adds parameter to turn on and off the upgrading step */
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "constraints/" CONSHDLR_NAME "/upgrade/%s", conshdlrname);
   (void) SCIPsnprintf(paramdesc, SCIP_MAXSTRLEN, "enable expression upgrading for constraint handler <%s>", conshdlrname);
   SCIP_CALL( SCIPaddBoolParam(scip,
         paramname, paramdesc,
         &exprconsupgrade->active, FALSE, active, NULL, NULL) );

   return SCIP_OKAY;
}

/** creates and captures a expr constraint
 *
 *  @note the constraint gets captured, hence at one point you have to release it using the method SCIPreleaseCons()
 */
SCIP_RETCODE SCIPcreateConsExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression of constraint (must not be NULL) */
   SCIP_Real             lhs,                /**< left hand side of constraint */
   SCIP_Real             rhs,                /**< right hand side of constraint */
   SCIP_Bool             initial,            /**< should the LP relaxation of constraint be in the initial LP?
                                              *   Usually set to TRUE. Set to FALSE for 'lazy constraints'. */
   SCIP_Bool             separate,           /**< should the constraint be separated during LP processing?
                                              *   Usually set to TRUE. */
   SCIP_Bool             enforce,            /**< should the constraint be enforced during node processing?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool             check,              /**< should the constraint be checked for feasibility?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool             propagate,          /**< should the constraint be propagated during node processing?
                                              *   Usually set to TRUE. */
   SCIP_Bool             local,              /**< is constraint only valid locally?
                                              *   Usually set to FALSE. Has to be set to TRUE, e.g., for branching constraints. */
   SCIP_Bool             modifiable,         /**< is constraint modifiable (subject to column generation)?
                                              *   Usually set to FALSE. In column generation applications, set to TRUE if pricing
                                              *   adds coefficients to this constraint. */
   SCIP_Bool             dynamic,            /**< is constraint subject to aging?
                                              *   Usually set to FALSE. Set to TRUE for own cuts which
                                              *   are separated as constraints. */
   SCIP_Bool             removable,          /**< should the relaxation be removed from the LP due to aging or cleanup?
                                              *   Usually set to FALSE. Set to TRUE for 'lazy constraints' and 'user cuts'. */
   SCIP_Bool             stickingatnode      /**< should the constraint always be kept at the node where it was added, even
                                              *   if it may be moved to a more global node?
                                              *   Usually set to FALSE. Set to TRUE to for constraints that represent node data. */
   )
{
   /* TODO: (optional) modify the definition of the SCIPcreateConsExpr() call, if you don't need all the information */

   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSDATA* consdata;

   assert(expr != NULL);

   /* find the expr constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   if( conshdlr == NULL )
   {
      SCIPerrorMessage("expr constraint handler not found\n");
      return SCIP_PLUGINNOTFOUND;
   }

   /* create constraint data */
   SCIP_CALL( SCIPallocClearBlockMemory(scip, &consdata) );
   consdata->expr = expr;
   consdata->lhs = lhs;
   consdata->rhs = rhs;

   /* capture expression */
   SCIPcaptureConsExprExpr(consdata->expr);

   /* create constraint */
   SCIP_CALL( SCIPcreateCons(scip, cons, name, conshdlr, consdata, initial, separate, enforce, check, propagate,
         local, modifiable, dynamic, removable, stickingatnode) );

   return SCIP_OKAY;
}

/** creates and captures a expr constraint with all its constraint flags set to their
 *  default values
 *
 *  @note the constraint gets captured, hence at one point you have to release it using the method SCIPreleaseCons()
 */
SCIP_RETCODE SCIPcreateConsExprBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression of constraint (must not be NULL) */
   SCIP_Real             lhs,                /**< left hand side of constraint */
   SCIP_Real             rhs                 /**< right hand side of constraint */
   )
{
   SCIP_CALL( SCIPcreateConsExpr(scip, cons, name, expr, lhs, rhs,
         TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   return SCIP_OKAY;
}

/** returns the expression of the given expression constraint */
SCIP_CONSEXPR_EXPR* SCIPgetExprConsExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(cons != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not expression\n");
      SCIPABORT();
      return NULL;  /*lint !e527*/
   }

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->expr;
}

/** gets the left hand side of an expression constraint */
SCIP_Real SCIPgetLhsConsExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) == 0);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->lhs;
}

/** gets the right hand side of an expression constraint */
SCIP_Real SCIPgetRhsConsExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) == 0);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->rhs;
}

/** returns an equivalent linear constraint if possible */
SCIP_RETCODE SCIPgetLinearConsExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint data */
   SCIP_CONS**           lincons             /**< buffer to store linear constraint data */
   )
{
   SCIP_CONSEXPR_EXPRHDLR* sumhdlr;
   SCIP_CONSEXPR_EXPRHDLR* varhdlr;
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSDATA* consdata;
   SCIP_CONSEXPR_EXPR* expr;
   SCIP_VAR** vars;
   SCIP_Real lhs;
   SCIP_Real rhs;
   int i;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(lincons != NULL);

   *lincons = NULL;
   expr = SCIPgetExprConsExpr(scip, cons);

   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);
   sumhdlr = SCIPgetConsExprExprHdlrSum(conshdlr);
   assert(sumhdlr != NULL);
   varhdlr = SCIPgetConsExprExprHdlrVar(conshdlr);
   assert(varhdlr != NULL);
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* not a linear constraint if the root expression is not a sum */
   if( expr == NULL || expr->exprhdlr != sumhdlr )
      return SCIP_OKAY;

   for( i = 0; i < SCIPgetConsExprExprNChildren(expr); ++i )
   {
      SCIP_CONSEXPR_EXPR* child = SCIPgetConsExprExprChildren(expr)[i];

      /* at least one child is not a variable -> not a linear constraint */
      if( child->exprhdlr != varhdlr )
         return SCIP_OKAY;
   }

   /* collect all variables */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, SCIPgetConsExprExprNChildren(expr)) );
   for( i = 0; i < SCIPgetConsExprExprNChildren(expr); ++i )
   {
      SCIP_CONSEXPR_EXPR* child = SCIPgetConsExprExprChildren(expr)[i];

      assert(child->exprhdlr == varhdlr);
      vars[i] = SCIPgetConsExprExprVarVar(child);
   }

   /* consider constant part of the sum expression */
   lhs = SCIPisInfinity(scip, -consdata->lhs) ? -SCIPinfinity(scip) : (consdata->lhs - SCIPgetConsExprExprSumConstant(expr));
   rhs = SCIPisInfinity(scip,  consdata->rhs) ?  SCIPinfinity(scip) : (consdata->rhs - SCIPgetConsExprExprSumConstant(expr));

   SCIP_CALL( SCIPcreateConsLinear(scip, lincons, SCIPconsGetName(cons),
         SCIPgetConsExprExprNChildren(expr), vars, SCIPgetConsExprExprSumCoefs(expr),
         lhs, rhs,
         SCIPconsIsInitial(cons), SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons),
         SCIPconsIsChecked(cons), SCIPconsIsPropagated(cons), SCIPconsIsLocal(cons),
         SCIPconsIsModifiable(cons), SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons),
         SCIPconsIsStickingAtNode(cons)) );

   /* free memory */
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}

/** returns a variable that appears linearly that may be decreased without making any other constraint infeasible */
SCIP_RETCODE SCIPgetLinvarMayDecreaseExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_CONS*            cons,               /**< expression constraint */
   SCIP_VAR**            var,                /**< pointer to store the variable */
   SCIP_Real*            coef                /**< pointer to store the coefficient */
   )
{
   SCIP_CONSDATA* consdata;

   assert(conshdlr != NULL);
   assert(cons != NULL);
   assert(var != NULL);
   assert(coef != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* check for a linear variable that can be increased or decreased without harming feasibility */
   consdataFindUnlockedLinearVar(scip, conshdlr, consdata);

   *var = consdata->linvardecr;
   *coef = consdata->linvardecrcoef;

   return SCIP_OKAY;
}

/** returns a variable that appears linearly that may be increased without making any other constraint infeasible */
SCIP_RETCODE SCIPgetLinvarMayIncreaseExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_CONS*            cons,               /**< expression constraint */
   SCIP_VAR**            var,                /**< pointer to store the variable */
   SCIP_Real*            coef                /**< pointer to store the coefficient */
   )
{
   SCIP_CONSDATA* consdata;

   assert(conshdlr != NULL);
   assert(cons != NULL);
   assert(var != NULL);
   assert(coef != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* check for a linear variable that can be increased or decreased without harming feasibility */
   consdataFindUnlockedLinearVar(scip, conshdlr, consdata);

   *var = consdata->linvarincr;
   *coef = consdata->linvarincrcoef;

   return SCIP_OKAY;
}


/** creates the nonlinearity handler and includes it into the expression constraint handler */
SCIP_RETCODE SCIPincludeConsExprNlhdlrBasic(
   SCIP*                       scip,         /**< SCIP data structure */
   SCIP_CONSHDLR*              conshdlr,     /**< expression constraint handler */
   SCIP_CONSEXPR_NLHDLR**      nlhdlr,       /**< buffer where to store nonlinear handler */
   const char*                 name,         /**< name of nonlinear handler (must not be NULL) */
   const char*                 desc,         /**< description of nonlinear handler (can be NULL) */
   int                         priority,     /**< priority of nonlinear handler */
   SCIP_DECL_CONSEXPR_NLHDLRDETECT((*detect)), /**< structure detection callback of nonlinear handler */
   SCIP_DECL_CONSEXPR_NLHDLREVALAUX((*evalaux)), /**< auxiliary evaluation callback of nonlinear handler */
   SCIP_CONSEXPR_NLHDLRDATA*   data          /**< data of nonlinear handler (can be NULL) */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   char paramname[SCIP_MAXSTRLEN];

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(nlhdlr != NULL);
   assert(name != NULL);
   assert(detect != NULL);
   assert(evalaux != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIP_CALL( SCIPallocClearMemory(scip, nlhdlr) );

   SCIP_CALL( SCIPduplicateMemoryArray(scip, &(*nlhdlr)->name, name, strlen(name)+1) );
   if( desc != NULL )
   {
      SCIP_CALL( SCIPduplicateMemoryArray(scip, &(*nlhdlr)->desc, desc, strlen(desc)+1) );
   }

   (*nlhdlr)->priority = priority;
   (*nlhdlr)->data = data;
   (*nlhdlr)->detect = detect;
   (*nlhdlr)->evalaux = evalaux;

   SCIP_CALL( SCIPcreateClock(scip, &(*nlhdlr)->detecttime) );
   SCIP_CALL( SCIPcreateClock(scip, &(*nlhdlr)->sepatime) );
   SCIP_CALL( SCIPcreateClock(scip, &(*nlhdlr)->proptime) );
   SCIP_CALL( SCIPcreateClock(scip, &(*nlhdlr)->intevaltime) );
   SCIP_CALL( SCIPcreateClock(scip, &(*nlhdlr)->reformulatetime) );

   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "constraints/expr/nlhdlr/%s/enabled", name);
   SCIP_CALL( SCIPaddBoolParam(scip, paramname, "should this nonlinear handler be used",
      &(*nlhdlr)->enabled, FALSE, TRUE, NULL, NULL) );

   ENSUREBLOCKMEMORYARRAYSIZE(scip, conshdlrdata->nlhdlrs, conshdlrdata->nlhdlrssize, conshdlrdata->nnlhdlrs+1);

   conshdlrdata->nlhdlrs[conshdlrdata->nnlhdlrs] = *nlhdlr;
   ++conshdlrdata->nnlhdlrs;

   /* sort nonlinear handlers by priority, in decreasing order
    * will happen in INIT, so only do when called late
    */
   if( SCIPgetStage(scip) >= SCIP_STAGE_INIT && conshdlrdata->nnlhdlrs > 1 )
      SCIPsortDownPtr((void**)conshdlrdata->nlhdlrs, nlhdlrCmp, conshdlrdata->nnlhdlrs);

   return SCIP_OKAY;
}

/** set the copy handler callback of a nonlinear handler */
void SCIPsetConsExprNlhdlrCopyHdlr(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLR*      nlhdlr,        /**< nonlinear handler */
   SCIP_DECL_CONSEXPR_NLHDLRCOPYHDLR((*copy)) /**< copy callback (can be NULL) */
   )
{
   assert(nlhdlr != NULL);

   nlhdlr->copyhdlr = copy;
}

/** set the nonlinear handler callback to free the nonlinear handler data */
void SCIPsetConsExprNlhdlrFreeHdlrData(
   SCIP*                      scip,              /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLR*      nlhdlr,            /**< nonlinear handler */
   SCIP_DECL_CONSEXPR_NLHDLRFREEHDLRDATA((*freehdlrdata)) /**< handler free callback (can be NULL) */
   )
{
   assert(nlhdlr != NULL);

   nlhdlr->freehdlrdata = freehdlrdata;
}

/** set the expression handler callback to free expression specific data of nonlinear handler */
void SCIPsetConsExprNlhdlrFreeExprData(
   SCIP*                      scip,              /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLR*      nlhdlr,            /**< nonlinear handler */
   SCIP_DECL_CONSEXPR_NLHDLRFREEEXPRDATA((*freeexprdata)) /**< nonlinear handler expression data free callback (can be NULL if data does not need to be freed) */
   )
{
   assert(nlhdlr != NULL);

   nlhdlr->freeexprdata = freeexprdata;
}

/** set the initialization and deinitialization callback of a nonlinear handler */
void SCIPsetConsExprNlhdlrInitExit(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLR*      nlhdlr,        /**< nonlinear handler */
   SCIP_DECL_CONSEXPR_NLHDLRINIT((*init)),   /**< initialization callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLREXIT((*exit_))    /**< deinitialization callback (can be NULL) */
   )
{
   assert(nlhdlr != NULL);

   nlhdlr->init = init;
   nlhdlr->exit = exit_;
}

/** set the reformulate callback of a nonlinear handler */
void SCIPsetConsExprNlhdlrReformulate(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLR*      nlhdlr,        /**< nonlinear handler */
   SCIP_DECL_CONSEXPR_NLHDLRREFORMULATE((*reformulate)) /**< reformulation callback */
   )
{
   assert(nlhdlr != NULL);

   nlhdlr->reformulate = reformulate;
}

/** set the propagation callbacks of a nonlinear handler */
void SCIPsetConsExprNlhdlrProp(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLR*      nlhdlr,        /**< nonlinear handler */
   SCIP_DECL_CONSEXPR_NLHDLRINTEVAL((*inteval)), /**< interval evaluation callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRREVERSEPROP((*reverseprop)) /**< reverse propagation callback (can be NULL) */
   )
{
   assert(nlhdlr != NULL);

   nlhdlr->inteval = inteval;
   nlhdlr->reverseprop = reverseprop;
}

/** set the separation callbacks of a nonlinear handler */
void SCIPsetConsExprNlhdlrSepa(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLR*      nlhdlr,        /**< nonlinear handler */
   SCIP_DECL_CONSEXPR_NLHDLRINITSEPA((*initsepa)), /**< separation initialization callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRSEPA((*sepa)),         /**< separation callback (can be NULL if estimate is not NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRESTIMATE((*estimate)), /**< estimation callback (can be NULL if sepa is not NULL) */
   SCIP_DECL_CONSEXPR_NLHDLREXITSEPA((*exitsepa))  /**< separation deinitialization callback (can be NULL) */
   )
{
   assert(nlhdlr != NULL);
   assert(sepa != NULL || estimate != NULL);

   nlhdlr->initsepa = initsepa;
   nlhdlr->sepa = sepa;
   nlhdlr->estimate = estimate;
   nlhdlr->exitsepa = exitsepa;
}

/** set the branching score callback of a nonlinear handler */
void SCIPsetConsExprNlhdlrBranchscore(
   SCIP*                      scip,          /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLR*      nlhdlr,        /**< nonlinear handler */
   SCIP_DECL_CONSEXPR_NLHDLRBRANCHSCORE((*branchscore)) /**< branching score callback */
   )
{
   assert(nlhdlr != NULL);

   nlhdlr->branchscore = branchscore;
}

/** gives name of nonlinear handler */
const char* SCIPgetConsExprNlhdlrName(
   SCIP_CONSEXPR_NLHDLR*      nlhdlr         /**< nonlinear handler */
   )
{
   assert(nlhdlr != NULL);

   return nlhdlr->name;
}

/** gives description of nonlinear handler, can be NULL */
const char* SCIPgetConsExprNlhdlrDesc(
   SCIP_CONSEXPR_NLHDLR*      nlhdlr         /**< nonlinear handler */
   )
{
   assert(nlhdlr != NULL);

   return nlhdlr->desc;
}

/** gives priority of nonlinear handler */
int SCIPgetConsExprNlhdlrPriority(
   SCIP_CONSEXPR_NLHDLR*      nlhdlr         /**< nonlinear handler */
   )
{
   assert(nlhdlr != NULL);

   return nlhdlr->priority;
}

/** returns a nonlinear handler of a given name (or NULL if not found) */
SCIP_CONSEXPR_NLHDLR* SCIPfindConsExprNlhdlr(
   SCIP_CONSHDLR*             conshdlr,      /**< expression constraint handler */
   const char*                name           /**< name of nonlinear handler */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   int h;

   assert(conshdlr != NULL);
   assert(name != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   for( h = 0; h < conshdlrdata->nnlhdlrs; ++h )
      if( strcmp(SCIPgetConsExprNlhdlrName(conshdlrdata->nlhdlrs[h]), name) == 0 )
         return conshdlrdata->nlhdlrs[h];

   return NULL;
}

/** gives handler data of nonlinear handler */
SCIP_CONSEXPR_NLHDLRDATA* SCIPgetConsExprNlhdlrData(
   SCIP_CONSEXPR_NLHDLR*      nlhdlr         /**< nonlinear handler */
   )
{
   assert(nlhdlr != NULL);

   return nlhdlr->data;
}

/** gives nonlinear handler expression data
 *
 * @return NULL if expr has not been detected by nlhdlr or nlhdlr did not store data
 */
SCIP_CONSEXPR_NLHDLREXPRDATA* SCIPgetConsExprNlhdlrExprData(
   SCIP_CONSEXPR_NLHDLR*      nlhdlr,        /**< nonlinear handler */
   SCIP_CONSEXPR_EXPR*        expr           /**< expression */
   )
{
   int e;

   for( e = 0; e < expr->nenfos; ++e )
   {
      if( expr->enfos[e]->nlhdlr == nlhdlr )
         return expr->enfos[e]->nlhdlrexprdata;
   }

   return NULL;
}

/** returns whether nonlinear handler implements the reformulation callback */
SCIP_Bool SCIPhasConsExprNlhdlrReformulate(
   SCIP_CONSEXPR_NLHDLR* nlhdlr              /**< nonlinear handler */
   )
{
   return nlhdlr->reformulate != NULL;
}

/** returns whether nonlinear handler implements the interval evaluation callback */
SCIP_Bool SCIPhasConsExprNlhdlrInteval(
   SCIP_CONSEXPR_NLHDLR* nlhdlr              /**< nonlinear handler */
   )
{
   return nlhdlr->inteval != NULL;
}

/** returns whether nonlinear handler implements the reverse propagation callback */
SCIP_Bool SCIPhasConsExprNlhdlrReverseProp(
   SCIP_CONSEXPR_NLHDLR* nlhdlr              /**< nonlinear handler */
   )
{
   return nlhdlr->reverseprop != NULL;
}

/** returns whether nonlinear handler implements the separation initialization callback */
SCIP_Bool SCIPhasConsExprNlhdlrInitSepa(
   SCIP_CONSEXPR_NLHDLR* nlhdlr              /**< nonlinear handler */
   )
{
   return nlhdlr->initsepa != NULL;
}

/** returns whether nonlinear handler implements the separation deinitialization callback */
SCIP_Bool SCIPhasConsExprNlhdlrExitSepa(
   SCIP_CONSEXPR_NLHDLR* nlhdlr              /**< nonlinear handler */
   )
{
   return nlhdlr->exitsepa != NULL;
}

/** returns whether nonlinear handler implements the separation callback */
SCIP_Bool SCIPhasConsExprNlhdlrSepa(
   SCIP_CONSEXPR_NLHDLR* nlhdlr              /**< nonlinear handler */
   )
{
   return nlhdlr->sepa != NULL;
}

/** returns whether nonlinear handler implements the estimator callback */
SCIP_Bool SCIPhasConsExprNlhdlrEstimate(
   SCIP_CONSEXPR_NLHDLR* nlhdlr              /**< nonlinear handler */
   )
{
   return nlhdlr->estimate != NULL;
}

/** call the detect callback of a nonlinear handler */
SCIP_DECL_CONSEXPR_NLHDLRDETECT(SCIPdetectConsExprNlhdlr)
{
   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(nlhdlr->detect != NULL);
   assert(nlhdlr->detecttime != NULL);
   assert(success != NULL);

   SCIP_CALL( SCIPstartClock(scip, nlhdlr->detecttime) );
   SCIP_CALL( nlhdlr->detect(scip, conshdlr, nlhdlr, expr, isroot, enforcemethods, enforcedbelow, enforcedabove, success, nlhdlrexprdata) );
   SCIP_CALL( SCIPstopClock(scip, nlhdlr->detecttime) );

   if( *success )
      ++nlhdlr->ndetections;

   return SCIP_OKAY;
}

/** calls the reformulation callback of a nonlinear handler */
SCIP_DECL_CONSEXPR_NLHDLRREFORMULATE(SCIPreformulateConsExprNlhdlr)
{
   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(nlhdlr != NULL);
   assert(nlhdlr->reformulatetime != NULL);

   *refexpr = NULL;

   if( nlhdlr->reformulate == NULL )
      return SCIP_OKAY;

   /* call reformulation callback */
   SCIP_CALL( SCIPstartClock(scip, nlhdlr->reformulatetime) );
   SCIP_CALL( nlhdlr->reformulate(scip, conshdlr, nlhdlr, expr, refexpr) );
   SCIP_CALL( SCIPstopClock(scip, nlhdlr->reformulatetime) );

   /* check whether reformulation was successful */
   if( *refexpr != NULL && *refexpr != expr )
      ++nlhdlr->nreformulates;

   return SCIP_OKAY;
}

/** call the auxiliary evaluation callback of a nonlinear handler */
SCIP_DECL_CONSEXPR_NLHDLREVALAUX(SCIPevalauxConsExprNlhdlr)
{
   assert(nlhdlr != NULL);
   assert(nlhdlr->evalaux != NULL);

   SCIP_CALL( nlhdlr->evalaux(scip, nlhdlr, expr, nlhdlrexprdata, auxvalue, sol) );

   return SCIP_OKAY;
}

/** calls the interval evaluation callback of a nonlinear handler */
SCIP_DECL_CONSEXPR_NLHDLRINTEVAL(SCIPintevalConsExprNlhdlr)
{
   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(nlhdlr->intevaltime != NULL);

   if( nlhdlr->inteval != NULL )
   {
      SCIP_CALL( SCIPstartClock(scip, nlhdlr->intevaltime) );
      SCIP_CALL( nlhdlr->inteval(scip, nlhdlr, expr, nlhdlrexprdata, interval, intevalvar, intevalvardata) );
      SCIP_CALL( SCIPstopClock(scip, nlhdlr->intevaltime) );
   }

   return SCIP_OKAY;
}

/** calls the reverse propagation callback of a nonlinear handler */
SCIP_DECL_CONSEXPR_NLHDLRREVERSEPROP(SCIPreversepropConsExprNlhdlr)
{
   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(nlhdlr->proptime != NULL);
   assert(infeasible != NULL);
   assert(nreductions != NULL);

   if( nlhdlr->reverseprop == NULL )
   {
      *infeasible = FALSE;
      *nreductions = 0;

      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPstartClock(scip, nlhdlr->proptime) );
   SCIP_CALL( nlhdlr->reverseprop(scip, nlhdlr, expr, nlhdlrexprdata, reversepropqueue, infeasible, nreductions, force) );
   SCIP_CALL( SCIPstopClock(scip, nlhdlr->proptime) );

   /* update statistics */
   nlhdlr->ndomreds += *nreductions;
   if( *infeasible )
      ++nlhdlr->ncutoffs;
   ++nlhdlr->npropcalls;

   return SCIP_OKAY;
}

/** calls the separation initialization callback of a nonlinear handler */
SCIP_DECL_CONSEXPR_NLHDLRINITSEPA(SCIPinitsepaConsExprNlhdlr)
{
   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(nlhdlr->sepatime != NULL);
   assert(infeasible != NULL);

   if( nlhdlr->initsepa == NULL )
   {
      *infeasible = FALSE;
      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPstartClock(scip, nlhdlr->sepatime) );
   SCIP_CALL( nlhdlr->initsepa(scip, conshdlr, nlhdlr, expr, nlhdlrexprdata, overestimate, underestimate, infeasible) );
   SCIP_CALL( SCIPstopClock(scip, nlhdlr->sepatime) );

   ++nlhdlr->nsepacalls;
   if( *infeasible )
      ++nlhdlr->ncutoffs;

   return SCIP_OKAY;
}

/** calls the separation deinitialization callback of a nonlinear handler */
SCIP_DECL_CONSEXPR_NLHDLREXITSEPA(SCIPexitsepaConsExprNlhdlr)
{
   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(nlhdlr->sepatime != NULL);

   if( nlhdlr->exitsepa != NULL )
   {
      SCIP_CALL( SCIPstartClock(scip, nlhdlr->sepatime) );
      SCIP_CALL( nlhdlr->exitsepa(scip, nlhdlr, expr, nlhdlrexprdata) );
      SCIP_CALL( SCIPstopClock(scip, nlhdlr->sepatime) );
   }

   return SCIP_OKAY;
}

/** calls the separation callback of a nonlinear handler */
SCIP_DECL_CONSEXPR_NLHDLRSEPA(SCIPsepaConsExprNlhdlr)
{
   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(nlhdlr->sepatime != NULL);
   assert(result != NULL);

   if( nlhdlr->sepa == NULL )
   {
      *result = SCIP_DIDNOTRUN;
      *ncuts = 0;
      return SCIP_OKAY;
   }

#ifndef NDEBUG
   /* check that auxvalue is correct by reevaluating */
   {
      SCIP_Real auxvaluetest;
      SCIP_CALL( SCIPevalauxConsExprNlhdlr(scip, nlhdlr, expr, nlhdlrexprdata, &auxvaluetest, sol) );
      assert(auxvalue == auxvaluetest);  /* we should get EXACTLY the same value from calling evalaux with the same solution as before */  /*lint !e777*/
   }
#endif

   SCIP_CALL( SCIPstartClock(scip, nlhdlr->sepatime) );
   SCIP_CALL( nlhdlr->sepa(scip, conshdlr, nlhdlr, expr, nlhdlrexprdata, sol, auxvalue, overestimate, mincutviolation, separated, result, ncuts) );
   SCIP_CALL( SCIPstopClock(scip, nlhdlr->sepatime) );

   /* update statistics */
   ++nlhdlr->nsepacalls;
   nlhdlr->ncutsfound += *ncuts;
   if( *result == SCIP_CUTOFF )
      ++nlhdlr->ncutoffs;

   return SCIP_OKAY;
}

/** calls the estimator callback of a nonlinear handler */
SCIP_DECL_CONSEXPR_NLHDLRESTIMATE(SCIPestimateConsExprNlhdlr)
{
   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(nlhdlr->sepatime != NULL);
   assert(success != NULL);

   if( nlhdlr->estimate == NULL )
   {
      *success = FALSE;
      return SCIP_OKAY;
   }

#ifndef NDEBUG
   /* check that auxvalue is correct by reevaluating */
   {
      SCIP_Real auxvaluetest;
      SCIP_CALL( SCIPevalauxConsExprNlhdlr(scip, nlhdlr, expr, nlhdlrexprdata, &auxvaluetest, sol) );
      assert(auxvalue == auxvaluetest);  /* we should get EXACTLY the same value from calling evalaux with the same solution as before */  /*lint !e777*/
   }
#endif

   SCIP_CALL( SCIPstartClock(scip, nlhdlr->sepatime) );
   SCIP_CALL( nlhdlr->estimate(scip, conshdlr, nlhdlr, expr, nlhdlrexprdata, sol, auxvalue, overestimate, targetvalue, rowprep, success) );
   SCIP_CALL( SCIPstopClock(scip, nlhdlr->sepatime) );

   /* update statistics */
   ++nlhdlr->nsepacalls;

   return SCIP_OKAY;
}

/** calls the nonlinear handler branching score callback */
SCIP_DECL_CONSEXPR_NLHDLRBRANCHSCORE(SCIPbranchscoreConsExprNlHdlr)
{
   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(success != NULL);

   if( nlhdlr->branchscore == NULL )
   {
      *success = FALSE;
      return SCIP_OKAY;
   }

#ifndef NDEBUG
   /* check that auxvalue is correct by reevaluating */
   {
      SCIP_Real auxvaluetest;
      SCIP_CALL( SCIPevalauxConsExprNlhdlr(scip, nlhdlr, expr, nlhdlrexprdata, &auxvaluetest, sol) );
      assert(auxvalue == auxvaluetest);  /* we should get EXACTLY the same value from calling evalaux with the same solution as before */  /*lint !e777*/
   }
#endif

   SCIP_CALL( nlhdlr->branchscore(scip, nlhdlr, expr, nlhdlrexprdata, sol, auxvalue, brscoretag, success) );

   if( *success )
      ++nlhdlr->nbranchscores;

   return SCIP_OKAY;
}

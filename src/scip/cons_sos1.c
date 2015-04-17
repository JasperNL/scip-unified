/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2015 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   cons_sos1.c
 * @brief  constraint handler for SOS type 1 constraints
 * @author Tobias Fischer
 * @author Marc Pfetsch
 *
 * A specially ordered set of type 1 (SOS1) is a sequence of variables such that at most one
 * variable is nonzero. The special case of two variables arises, for instance, from equilibrium or
 * complementary conditions like \f$x \cdot y = 0\f$. Note that it is in principle allowed that a
 * variables appears twice, but it then can be fixed to 0.
 *
 * This implementation of this constraint handler is based on classical ideas, see e.g.@n
 *  "Special Facilities in General Mathematical Programming System for
 *  Non-Convex Problems Using Ordered Sets of Variables"@n
 *  E. Beale and J. Tomlin, Proc. 5th IFORS Conference, 447-454 (1970)
 *
 *
 * The order of the variables is determined as follows:
 *
 * - If the constraint is created with SCIPcreateConsSOS1() and weights are given, the weights
 *   determine the order (decreasing weights). Additional variables can be added with
 *   SCIPaddVarSOS1(), which adds a variable with given weight.
 *
 * - If an empty constraint is created and then variables are added with SCIPaddVarSOS1(), weights
 *   are needed and stored.
 *
 * - All other calls ignore the weights, i.e., if a nonempty constraint is created or variables are
 *   added with SCIPappendVarSOS1().
 *
 * The validity of the SOS1 constraint can be enforced by different branching rules:
 *
 * - If classical SOS branching is used, branching is performed on only one SOS1 constraint. Depending on the parameters,
 *   there are two ways to choose this branching constraint. Either the constraint with the most number of nonzeros
 *   or the one with the largest nonzero-variable weight. The later version allows the user to specify
 *   an order for the branching importance of the constraints. Constraint branching can also be turned off.
 *
 * - Another way is to branch on the neighborhood of a single variable @p i, i.e., in one branch \f$x_i\f$ is fixed to zero
 *   and in the other its neighbors.
 *
 * - If bipartite branching is used, then we branch using complete bipartite subgraphs.
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "scip/cons_sos1.h"
#include "scip/cons_linear.h"
#include "scip/cons_setppc.h"
#include "scip/pub_misc.h"
#include "scip/misc.h"
#include "scip/struct_misc.h"
#include "tclique/tclique.h"
#include <string.h>
#include <ctype.h>


/* constraint handler properties */
#define CONSHDLR_NAME          "SOS1"
#define CONSHDLR_DESC          "SOS1 constraint handler"
#define CONSHDLR_SEPAPRIORITY   -900000 /**< priority of the constraint handler for separation */
#define CONSHDLR_ENFOPRIORITY       100 /**< priority of the constraint handler for constraint enforcing */
#define CONSHDLR_CHECKPRIORITY      -10 /**< priority of the constraint handler for checking feasibility */
#define CONSHDLR_SEPAFREQ            10 /**< frequency for separating cuts; zero means to separate only in the root node */
#define CONSHDLR_PROPFREQ             1 /**< frequency for propagating domains; zero means only preprocessing propagation */
#define CONSHDLR_EAGERFREQ          100 /**< frequency for using all instead of only the useful constraints in separation,
                                         *   propagation and enforcement, -1 for no eager evaluations, 0 for first only */
#define CONSHDLR_MAXPREROUNDS        -1 /**< maximal number of presolving rounds the constraint handler participates in (-1: no limit) */
#define CONSHDLR_DELAYSEPA        FALSE /**< should separation method be delayed, if other separators found cuts? */
#define CONSHDLR_DELAYPROP        FALSE /**< should propagation method be delayed, if other propagators found reductions? */
#define CONSHDLR_DELAYPRESOL       TRUE /**< should presolving method be delayed, if other presolvers found reductions? */
#define CONSHDLR_NEEDSCONS         TRUE /**< should the constraint handler be skipped, if no constraints are available? */

/* propagation */
#define DEFAULT_CONFLICTPROP      TRUE /**< whether to use conflict graph propagation */
#define DEFAULT_SOSCONSPROP      FALSE /**< whether to use SOS1 constraint propagation */

/* separation */
#define DEFAULT_SEPAFROMSOS1      FALSE /**< if TRUE separate bound inequalities from initial SOS1 constraints */
#define DEFAULT_SEPAFROMGRAPH      TRUE /**< if TRUE separate bound inequalities from the conflict graph */
#define DEFAULT_BOUNDCUTSDEPTH       40 /**< node depth of separating bound cuts (-1: no limit) */
#define DEFAULT_MAXBOUNDCUTS         50 /**< maximal number of bound cuts separated per branching node */
#define DEFAULT_MAXBOUNDCUTSROOT    150 /**< maximal number of bound cuts separated per iteration in the root node */
#define DEFAULT_STRTHENBOUNDCUTS   TRUE /**< if TRUE then bound cuts are strengthened in case bound variables are available */

#define CONSHDLR_PROP_TIMING       SCIP_PROPTIMING_BEFORELP

/* event handler properties */
#define EVENTHDLR_NAME         "SOS1"
#define EVENTHDLR_DESC         "bound change event handler for SOS1 constraints"


/** constraint data for SOS1 constraints */
struct SCIP_ConsData
{
   int                   nvars;              /**< number of variables in the constraint */
   int                   maxvars;            /**< maximal number of variables (= size of storage) */
   int                   nfixednonzeros;     /**< number of variables fixed to be nonzero */
   SCIP_Bool             local;              /**< TRUE if constraint is only valid locally */
   SCIP_VAR**            vars;               /**< variables in constraint */
   SCIP_ROW*             rowlb;              /**< row corresponding to lower bounds, or NULL if not yet created */
   SCIP_ROW*             rowub;              /**< row corresponding to upper bounds, or NULL if not yet created */
   SCIP_Real*            weights;            /**< weights determining the order (ascending), or NULL if not used */
};


/** node data of a given node in the conflict graph */
struct SCIP_NodeData
{
   SCIP_VAR*             var;                /**< variable belonging to node */
   SCIP_VAR*             lbboundvar;         /**< bound variable @p z from constraint \f$x \geq \mu \cdot z\f$ (or NULL if not existent) */
   SCIP_VAR*             ubboundvar;         /**< bound variable @p z from constraint \f$x \leq \mu \cdot z\f$ (or NULL if not existent) */
   SCIP_Real             lbboundcoef;        /**< value \f$\mu\f$ from constraint \f$x \geq \mu z \f$ (0.0 if not existent) */
   SCIP_Real             ubboundcoef;        /**< value \f$\mu\f$ from constraint \f$x \leq \mu z \f$ (0.0 if not existent) */
   SCIP_Bool             lbboundcomp;        /**< TRUE if the nodes from the connected component of the conflict graph the given node belongs to
                                              *   all have the same lower bound variable */
   SCIP_Bool             ubboundcomp;        /**< TRUE if the nodes from the connected component of the conflict graph the given node belongs to
                                              *   all have the same lower bound variable */
};
typedef struct SCIP_NodeData SCIP_NODEDATA;


/** tclique data for bound cut generation */
struct TCLIQUE_Data
{
   SCIP*                 scip;               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr;           /**< SOS1 constraint handler */
   SCIP_DIGRAPH*         conflictgraph;      /**< conflict graph */
   SCIP_SOL*             sol;                /**< LP solution to be separated (or NULL) */
   SCIP_Real             scaleval;           /**< factor for scaling weights */
   int                   ncuts;              /**< number of bound cuts found in this iteration */
   int                   nboundcuts;         /**< number of bound cuts found so far */
   int                   maxboundcuts;       /**< maximal number of clique cuts separated per separation round (-1: no limit) */
   SCIP_Bool             strthenboundcuts;   /**< if TRUE then bound cuts are strengthened in case bound variables are available */
};


/** SOS1 constraint handler data */
struct SCIP_ConshdlrData
{
   /* conflict graph */
   SCIP_DIGRAPH*         conflictgraph;      /**< conflict graph */
   SCIP_DIGRAPH*         localconflicts;     /**< local conflicts */
   SCIP_Bool             isconflocal;        /**< if TRUE then local conflicts are present and conflict graph has to be updated for each node */
   SCIP_HASHMAP*         varhash;            /**< hash map from variable to node in the conflict graph */
   int                   nsos1vars;          /**< number of problem variables that are involved in at least one SOS1 constraint */
   /* propagation */
   SCIP_Bool             conflictprop;       /**< whether to use conflict graph propagation */
   SCIP_Bool             sosconsprop;        /**< whether to use SOS1 constraint propagation */
   /* branching */
   SCIP_Bool             branchsos;          /**< Branch on SOS condition in enforcing? */
   SCIP_Bool             branchnonzeros;     /**< Branch on SOS cons. with most number of nonzeros? */
   SCIP_Bool             branchweight;       /**< Branch on SOS cons. with highest nonzero-variable weight for branching - needs branchnonzeros to be false */
   /* separation */
   SCIP_Bool             sepafromsos1;       /**< if TRUE separate bound inequalities from initial SOS1 constraints */
   SCIP_Bool             sepafromgraph;      /**< if TRUE separate bound inequalities from the conflict graph */
   TCLIQUE_GRAPH*        tcliquegraph;       /**< tclique graph data structure */
   TCLIQUE_DATA*         tcliquedata;        /**< tclique data */
   int                   boundcutsdepth;     /**< node depth of separating bound cuts (-1: no limit) */
   int                   maxboundcuts;       /**< maximal number of bound cuts separated per branching node */
   int                   maxboundcutsroot;   /**< maximal number of bound cuts separated per iteration in the root node */
   int                   nboundcuts;         /**< number of bound cuts found so far */
   SCIP_Bool             strthenboundcuts;   /**< if TRUE then bound cuts are strengthened in case bound variables are available */
   /* event handler */
   SCIP_EVENTHDLR*       eventhdlr;          /**< event handler for bound change events */
};


/** fix variable in given node to 0 or add constraint if variable is multi-aggregated */
static
SCIP_RETCODE fixVariableZeroNode(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_VAR*             var,                /**< variable to be fixed to 0*/
   SCIP_NODE*            node,               /**< node */
   SCIP_Bool*            infeasible          /**< if fixing is infeasible */
   )
{
   /* if variable cannot be nonzero */
   *infeasible = FALSE;
   if ( SCIPisFeasPositive(scip, SCIPvarGetLbLocal(var)) || SCIPisFeasNegative(scip, SCIPvarGetUbLocal(var)) )
   {
      *infeasible = TRUE;
      return SCIP_OKAY;
   }

   /* if variable is multi-aggregated */
   if ( SCIPvarGetStatus(var) == SCIP_VARSTATUS_MULTAGGR )
   {
      SCIP_CONS* cons;
      SCIP_Real val;

      val = 1.0;

      if ( ! SCIPisFeasZero(scip, SCIPvarGetLbLocal(var)) || ! SCIPisFeasZero(scip, SCIPvarGetUbLocal(var)) )
      {
         SCIPdebugMessage("creating constraint to force multi-aggregated variable <%s> to 0.\n", SCIPvarGetName(var));
         /* we have to insert a local constraint var = 0 */
         SCIP_CALL( SCIPcreateConsLinear(scip, &cons, "branch", 1, &var, &val, 0.0, 0.0, TRUE, TRUE, TRUE, TRUE, TRUE,
               TRUE, FALSE, FALSE, FALSE, FALSE) );
         SCIP_CALL( SCIPaddConsNode(scip, node, cons, NULL) );
         SCIP_CALL( SCIPreleaseCons(scip, &cons) );
      }
   }
   else
   {
      if ( ! SCIPisFeasZero(scip, SCIPvarGetLbLocal(var)) )
         SCIP_CALL( SCIPchgVarLbNode(scip, node, var, 0.0) );
      if ( ! SCIPisFeasZero(scip, SCIPvarGetUbLocal(var)) )
         SCIP_CALL( SCIPchgVarUbNode(scip, node, var, 0.0) );
   }

   return SCIP_OKAY;
}


/** fix variable in local node to 0, and return whether the operation was feasible
 *
 *  @note We do not add a linear constraint if the variable is multi-aggregated as in
 *  fixVariableZeroNode(), since this would be too time consuming.
 */
static
SCIP_RETCODE inferVariableZero(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_VAR*             var,                /**< variable to be fixed to 0*/
   SCIP_CONS*            cons,               /**< constraint */
   int                   inferinfo,          /**< info for reverse prop. */
   SCIP_Bool*            infeasible,         /**< if fixing is infeasible */
   SCIP_Bool*            tightened,          /**< if fixing was performed */
   SCIP_Bool*            success             /**< whether fixing was successful, i.e., variable is not multi-aggregated */
   )
{
   *infeasible = FALSE;
   *tightened = FALSE;
   *success = FALSE;

   /* if variable cannot be nonzero */
   if ( SCIPisFeasPositive(scip, SCIPvarGetLbLocal(var)) || SCIPisFeasNegative(scip, SCIPvarGetUbLocal(var)) )
   {
      *infeasible = TRUE;
      return SCIP_OKAY;
   }

   /* directly fix variable if it is not multi-aggregated */
   if ( SCIPvarGetStatus(var) != SCIP_VARSTATUS_MULTAGGR )
   {
      SCIP_Bool tighten;

      /* fix lower bound */
      SCIP_CALL( SCIPinferVarLbCons(scip, var, 0.0, cons, inferinfo, FALSE, infeasible, &tighten) );
      *tightened = *tightened || tighten;

      /* fix upper bound */
      SCIP_CALL( SCIPinferVarUbCons(scip, var, 0.0, cons, inferinfo, FALSE, infeasible, &tighten) );
      *tightened = *tightened || tighten;

      *success = TRUE;
   }

   return SCIP_OKAY;
}


/** add lock on variable */
static
SCIP_RETCODE lockVariableSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_VAR*             var                 /**< variable */
   )
{
   assert( scip != NULL );
   assert( cons != NULL );
   assert( var != NULL );

   /* rounding down == bad if lb < 0, rounding up == bad if ub > 0 */
   SCIP_CALL( SCIPlockVarCons(scip, var, cons, SCIPisFeasNegative(scip, SCIPvarGetLbLocal(var)), SCIPisFeasPositive(scip, SCIPvarGetUbLocal(var))) );

   return SCIP_OKAY;
}


/* remove lock on variable */
static
SCIP_RETCODE unlockVariableSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_VAR*             var                 /**< variable */
   )
{
   assert( scip != NULL );
   assert( cons != NULL );
   assert( var != NULL );

   /* rounding down == bad if lb < 0, rounding up == bad if ub > 0 */
   SCIP_CALL( SCIPunlockVarCons(scip, var, cons, SCIPisFeasNegative(scip, SCIPvarGetLbLocal(var)), SCIPisFeasPositive(scip, SCIPvarGetUbLocal(var))) );

   return SCIP_OKAY;
}


/** ensures that the vars and weights array can store at least num entries */
static
SCIP_RETCODE consdataEnsurevarsSizeSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSDATA*        consdata,           /**< constraint data */
   int                   num,                /**< minimum number of entries to store */
   SCIP_Bool             reserveWeights      /**< whether the weights array is handled */
   )
{
   assert( consdata != NULL );
   assert( consdata->nvars <= consdata->maxvars );

   if ( num > consdata->maxvars )
   {
      int newsize;

      newsize = SCIPcalcMemGrowSize(scip, num);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &consdata->vars, consdata->maxvars, newsize) );
      if ( reserveWeights )
         SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &consdata->weights, consdata->maxvars, newsize) );
      consdata->maxvars = newsize;
   }
   assert( num <= consdata->maxvars );

   return SCIP_OKAY;
}


/** handle new variable */
static
SCIP_RETCODE handleNewVariableSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_CONSDATA*        consdata,           /**< constraint data */
   SCIP_VAR*             var,                /**< variable */
   SCIP_Bool             transformed         /**< whether original variable was transformed */
   )
{
   assert( scip != NULL );
   assert( cons != NULL );
   assert( consdata != NULL );
   assert( var != NULL );

   /* if we are in transformed problem, catch the variable's events */
   if ( transformed )
   {
      SCIP_CONSHDLR* conshdlr;
      SCIP_CONSHDLRDATA* conshdlrdata;

      /* get event handler */
      conshdlr = SCIPconsGetHdlr(cons);
      conshdlrdata = SCIPconshdlrGetData(conshdlr);
      assert( conshdlrdata != NULL );
      assert( conshdlrdata->eventhdlr != NULL );

      /* catch bound change events of variable */
      SCIP_CALL( SCIPcatchVarEvent(scip, var, SCIP_EVENTTYPE_BOUNDCHANGED, conshdlrdata->eventhdlr,
            (SCIP_EVENTDATA*)consdata, NULL) );

      /* if the variable if fixed to nonzero */
      assert( consdata->nfixednonzeros >= 0 );
      if ( SCIPisFeasPositive(scip, SCIPvarGetLbLocal(var)) || SCIPisFeasNegative(scip, SCIPvarGetUbLocal(var)) )
         ++consdata->nfixednonzeros;
   }

   /* install the rounding locks for the new variable */
   SCIP_CALL( lockVariableSOS1(scip, cons, var) );

   /* branching on multiaggregated variables does not seem to work well, so avoid it */
   SCIP_CALL( SCIPmarkDoNotMultaggrVar(scip, var) );

   /* add the new coefficient to the upper bound LP row, if necessary */
   if ( consdata->rowub != NULL && ! SCIPisInfinity(scip, SCIPvarGetUbGlobal(var)) && ! SCIPisZero(scip, SCIPvarGetUbGlobal(var)) )
   {
      SCIP_CALL( SCIPaddVarToRow(scip, consdata->rowub, var, 1.0/SCIPvarGetUbGlobal(var)) );
   }

   /* add the new coefficient to the lower bound LP row, if necessary */
   if ( consdata->rowlb != NULL && ! SCIPisInfinity(scip, SCIPvarGetLbGlobal(var)) && ! SCIPisZero(scip, SCIPvarGetLbGlobal(var)) )
   {
      SCIP_CALL( SCIPaddVarToRow(scip, consdata->rowlb, var, 1.0/SCIPvarGetLbGlobal(var)) );
   }

   return SCIP_OKAY;
}


/** adds a variable to an SOS1 constraint, at position given by weight - ascending order */
static
SCIP_RETCODE addVarSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_VAR*             var,                /**< variable to add to the constraint */
   SCIP_Real             weight              /**< weight to determine position */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_Bool transformed;
   int pos;
   int j;

   assert( var != NULL );
   assert( cons != NULL );

   consdata = SCIPconsGetData(cons);
   assert( consdata != NULL );

   if ( consdata->weights == NULL && consdata->maxvars > 0 )
   {
      SCIPerrorMessage("cannot add variable to SOS1 constraint <%s> that does not contain weights.\n", SCIPconsGetName(cons));
      return SCIP_INVALIDCALL;
   }

   /* are we in the transformed problem? */
   transformed = SCIPconsIsTransformed(cons);

   /* always use transformed variables in transformed constraints */
   if ( transformed )
   {
      SCIP_CALL( SCIPgetTransformedVar(scip, var, &var) );
   }
   assert( var != NULL );
   assert( transformed == SCIPvarIsTransformed(var) );

   SCIP_CALL( consdataEnsurevarsSizeSOS1(scip, consdata, consdata->nvars + 1, TRUE) );
   assert( consdata->weights != NULL );
   assert( consdata->maxvars >= consdata->nvars+1 );

   /* find variable position */
   for (pos = 0; pos < consdata->nvars; ++pos)
   {
      if ( consdata->weights[pos] > weight )
         break;
   }
   assert( 0 <= pos && pos <= consdata->nvars );

   /* move other variables, if necessary */
   for (j = consdata->nvars; j > pos; --j)
   {
      consdata->vars[j] = consdata->vars[j-1];
      consdata->weights[j] = consdata->weights[j-1];
   }

   /* insert variable */
   consdata->vars[pos] = var;
   consdata->weights[pos] = weight;
   ++consdata->nvars;

   /* handle the new variable */
   SCIP_CALL( handleNewVariableSOS1(scip, cons, consdata, var, transformed) );

   return SCIP_OKAY;
}


/** appends a variable to an SOS1 constraint */
static
SCIP_RETCODE appendVarSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_VAR*             var                 /**< variable to add to the constraint */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_Bool transformed;

   assert( var != NULL );
   assert( cons != NULL );

   consdata = SCIPconsGetData(cons);
   assert( consdata != NULL );

   /* are we in the transformed problem? */
   transformed = SCIPconsIsTransformed(cons);

   /* always use transformed variables in transformed constraints */
   if ( transformed )
   {
      SCIP_CALL( SCIPgetTransformedVar(scip, var, &var) );
   }
   assert( var != NULL );
   assert( transformed == SCIPvarIsTransformed(var) );

   SCIP_CALL( consdataEnsurevarsSizeSOS1(scip, consdata, consdata->nvars + 1, FALSE) );

   /* insert variable */
   consdata->vars[consdata->nvars] = var;
   assert( consdata->weights != NULL || consdata->nvars > 0 );
   if ( consdata->weights != NULL && consdata->nvars > 0 )
      consdata->weights[consdata->nvars] = consdata->weights[consdata->nvars-1] + 1.0;
   ++consdata->nvars;

   /* handle the new variable */
   SCIP_CALL( handleNewVariableSOS1(scip, cons, consdata, var, transformed) );

   return SCIP_OKAY;
}


/** deletes a variable of an SOS1 constraint */
static
SCIP_RETCODE deleteVarSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_CONSDATA*        consdata,           /**< constraint data */
   SCIP_EVENTHDLR*       eventhdlr,          /**< corresponding event handler */
   int                   pos                 /**< position of variable in array */
   )
{
   int j;

   assert( 0 <= pos && pos < consdata->nvars );

   /* remove lock of variable */
   SCIP_CALL( unlockVariableSOS1(scip, cons, consdata->vars[pos]) );

   /* drop events on variable */
   SCIP_CALL( SCIPdropVarEvent(scip, consdata->vars[pos], SCIP_EVENTTYPE_BOUNDCHANGED, eventhdlr, (SCIP_EVENTDATA*)consdata, -1) );

   /* delete variable - need to copy since order is important */
   for (j = pos; j < consdata->nvars-1; ++j)
   {
      consdata->vars[j] = consdata->vars[j+1]; /*lint !e679*/
      if ( consdata->weights != NULL )
         consdata->weights[j] = consdata->weights[j+1]; /*lint !e679*/
   }
   --consdata->nvars;

   return SCIP_OKAY;
}


/** perform one presolving round
 *
 *  We perform the following presolving steps.
 *
 *  - If the bounds of some variable force it to be nonzero, we can
 *    fix all other variables to zero and remove the SOS1 constraints
 *    that contain it.
 *  - If a variable is fixed to zero, we can remove the variable.
 *  - If a variable appears twice, it can be fixed to 0.
 *  - We substitute appregated variables.
 */
static
SCIP_RETCODE presolRoundSOS1(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_CONSDATA*        consdata,           /**< constraint data */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler */
   SCIP_Bool*            cutoff,             /**< whether a cutoff happened */
   SCIP_Bool*            success,            /**< whether we performed a successful reduction */
   int*                  ndelconss,          /**< number of deleted constraints */
   int*                  nupgdconss,         /**< number of upgraded constraints */
   int*                  nfixedvars,         /**< number of fixed variables */
   int*                  nremovedvars        /**< number of variables removed */
   )
{
   SCIP_VAR** vars;
   SCIP_Bool allvarsbinary;
   SCIP_Bool infeasible;
   SCIP_Bool fixed;
   int nfixednonzeros;
   int lastFixedNonzero;
   int j;

   assert( scip != NULL );
   assert( cons != NULL );
   assert( consdata != NULL );
   assert( eventhdlr != NULL );
   assert( cutoff != NULL );
   assert( success != NULL );
   assert( ndelconss != NULL );
   assert( nfixedvars != NULL );
   assert( nremovedvars != NULL );

   *cutoff = FALSE;
   *success = FALSE;

   SCIPdebugMessage("Presolving SOS1 constraint <%s>.\n", SCIPconsGetName(cons) );

   j = 0;
   nfixednonzeros = 0;
   lastFixedNonzero = -1;
   allvarsbinary = TRUE;
   vars = consdata->vars;

   /* check for variables fixed to 0 and bounds that fix a variable to be nonzero */
   while ( j < consdata->nvars )
   {
      int l;
      SCIP_VAR* var;
      SCIP_Real lb;
      SCIP_Real ub;
      SCIP_Real scalar;
      SCIP_Real constant;

      scalar = 1.0;
      constant = 0.0;

      /* check for aggregation: if the constant is zero the variable is zero iff the aggregated
       * variable is 0 */
      var = vars[j];
      SCIP_CALL( SCIPgetProbvarSum(scip, &var, &scalar, &constant) );

      /* if constant is zero and we get a different variable, substitute variable */
      if ( SCIPisZero(scip, constant) && ! SCIPisZero(scip, scalar) && var != vars[j] )
      {
         SCIPdebugMessage("substituted variable <%s> by <%s>.\n", SCIPvarGetName(vars[j]), SCIPvarGetName(var));
         SCIP_CALL( SCIPdropVarEvent(scip, consdata->vars[j], SCIP_EVENTTYPE_BOUNDCHANGED, eventhdlr, (SCIP_EVENTDATA*)consdata, -1) );
         SCIP_CALL( SCIPcatchVarEvent(scip, var, SCIP_EVENTTYPE_BOUNDCHANGED, eventhdlr, (SCIP_EVENTDATA*)consdata, NULL) );

         /* change the rounding locks */
         SCIP_CALL( unlockVariableSOS1(scip, cons, consdata->vars[j]) );
         SCIP_CALL( lockVariableSOS1(scip, cons, var) );

         vars[j] = var;
      }

      /* check whether the variable appears again later */
      for (l = j+1; l < consdata->nvars; ++l)
      {
         /* if variable appeared before, we can fix it to 0 and remove it */
         if ( vars[j] == vars[l] )
         {
            SCIPdebugMessage("variable <%s> appears twice in constraint, fixing it to 0.\n", SCIPvarGetName(vars[j]));
            SCIP_CALL( SCIPfixVar(scip, vars[j], 0.0, &infeasible, &fixed) );

            if ( infeasible )
            {
               *cutoff = TRUE;
               return SCIP_OKAY;
            }
            if ( fixed )
               ++(*nfixedvars);
         }
      }

      /* get bounds */
      lb = SCIPvarGetLbLocal(vars[j]);
      ub = SCIPvarGetUbLocal(vars[j]);

      /* if the variable if fixed to nonzero */
      if ( SCIPisFeasPositive(scip, lb) || SCIPisFeasNegative(scip, ub) )
      {
         ++nfixednonzeros;
         lastFixedNonzero = j;
      }

      /* if the variable is fixed to 0 */
      if ( SCIPisFeasZero(scip, lb) && SCIPisFeasZero(scip, ub) )
      {
         SCIPdebugMessage("deleting variable <%s> fixed to 0.\n", SCIPvarGetName(vars[j]));
         SCIP_CALL( deleteVarSOS1(scip, cons, consdata, eventhdlr, j) );
         ++(*nremovedvars);
      }
      else
      {
         /* check whether all variables are binary */
         if ( ! SCIPvarIsBinary(vars[j]) )
            allvarsbinary = FALSE;

         ++j;
      }
   }

   /* if the number of variables is less than 2 */
   if ( consdata->nvars < 2 )
   {
      SCIPdebugMessage("Deleting SOS1 constraint <%s> with < 2 variables.\n", SCIPconsGetName(cons));

      /* delete constraint */
      assert( ! SCIPconsIsModifiable(cons) );
      SCIP_CALL( SCIPdelCons(scip, cons) );
      ++(*ndelconss);
      *success = TRUE;
      return SCIP_OKAY;
   }

   /* if more than one variable are fixed to be nonzero, we are infeasible */
   if ( nfixednonzeros > 1 )
   {
      SCIPdebugMessage("The problem is infeasible: more than one variable has bounds that keep it from being 0.\n");
      assert( lastFixedNonzero >= 0 );
      *cutoff = TRUE;
      return SCIP_OKAY;
   }

   /* if there is exactly one fixed nonzero variable */
   if ( nfixednonzeros == 1 )
   {
      assert( lastFixedNonzero >= 0 );

      /* fix all other variables to zero */
      for (j = 0; j < consdata->nvars; ++j)
      {
         if ( j != lastFixedNonzero )
         {
            SCIP_CALL( SCIPfixVar(scip, vars[j], 0.0, &infeasible, &fixed) );
            assert( ! infeasible );
            if ( fixed )
               ++(*nfixedvars);
         }
      }

      SCIPdebugMessage("Deleting redundant SOS1 constraint <%s> with one variable.\n", SCIPconsGetName(cons));

      /* delete original constraint */
      assert( ! SCIPconsIsModifiable(cons) );
      SCIP_CALL( SCIPdelCons(scip, cons) );
      ++(*ndelconss);
      *success = TRUE;
   }
   /* note: there is no need to update consdata->nfixednonzeros, since the constraint is deleted as soon nfixednonzeros > 0. */
   else
   {
      /* if all variables are binary create a set packing constraint */
      if ( allvarsbinary )
      {
         SCIP_CONS* setpackcons;

         /* create, add, and release the logicor constraint */
         SCIP_CALL( SCIPcreateConsSetpack(scip, &setpackcons, SCIPconsGetName(cons), consdata->nvars, consdata->vars,
               SCIPconsIsInitial(cons), SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons), SCIPconsIsChecked(cons),
               SCIPconsIsPropagated(cons), SCIPconsIsLocal(cons), SCIPconsIsModifiable(cons), SCIPconsIsDynamic(cons), 
               SCIPconsIsRemovable(cons), SCIPconsIsStickingAtNode(cons)) );
         SCIP_CALL( SCIPaddCons(scip, setpackcons) );
         SCIP_CALL( SCIPreleaseCons(scip, &setpackcons) );

         SCIPdebugMessage("Upgrading SOS1 constraint <%s> to set packing constraint.\n", SCIPconsGetName(cons));

         /* remove the SOS1 constraint globally */
         assert( ! SCIPconsIsModifiable(cons) );
         SCIP_CALL( SCIPdelCons(scip, cons) );
         ++(*nupgdconss);
         *success = TRUE;
      }
   }

   return SCIP_OKAY;
}


/** propagate variables */
static
SCIP_RETCODE propSOS1(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_CONSDATA*        consdata,           /**< constraint data */
   SCIP_Bool*            cutoff,             /**< whether a cutoff happened */
   int*                  ngen                /**< pointer to incremental counter for domain changes */
   )
{
   assert( scip != NULL );
   assert( cons != NULL );
   assert( consdata != NULL );
   assert( cutoff != NULL );
   assert( ngen != NULL );

   *cutoff = FALSE;

   /* if more than one variable is fixed to be nonzero */
   if ( consdata->nfixednonzeros > 1 )
   {
      SCIPdebugMessage("the node is infeasible, more than 1 variable is fixed to be nonzero.\n");
      SCIP_CALL( SCIPresetConsAge(scip, cons) );
      *cutoff = TRUE;
      return SCIP_OKAY;
   }

   /* if exactly one variable is fixed to be nonzero */
   if ( consdata->nfixednonzeros == 1 )
   {
      SCIP_VAR** vars;
      SCIP_Bool infeasible;
      SCIP_Bool tightened;
      SCIP_Bool success;
      SCIP_Bool allVarFixed;
      int firstFixedNonzero;
      int ngenold;
      int nvars;
      int j;

      firstFixedNonzero = -1;
      nvars = consdata->nvars;
      vars = consdata->vars;
      assert( vars != NULL );
      ngenold = *ngen;

      /* search nonzero variable - is needed for propinfo */
      for (j = 0; j < nvars; ++j)
      {
         if ( SCIPisFeasPositive(scip, SCIPvarGetLbLocal(vars[j])) || SCIPisFeasNegative(scip, SCIPvarGetUbLocal(vars[j])) )
         {
            firstFixedNonzero = j;
            break;
         }
      }
      assert( firstFixedNonzero >= 0 );

      SCIPdebugMessage("variable <%s> is fixed nonzero, fixing other variables to 0.\n", SCIPvarGetName(vars[firstFixedNonzero]));

      /* fix variables before firstFixedNonzero to 0 */
      allVarFixed = TRUE;
      for (j = 0; j < firstFixedNonzero; ++j)
      {
         /* fix variable */
         SCIP_CALL( inferVariableZero(scip, vars[j], cons, firstFixedNonzero, &infeasible, &tightened, &success) );
         assert( ! infeasible );
         allVarFixed = allVarFixed && success;
         if ( tightened )
            ++(*ngen);
      }

      /* fix variables after firstFixedNonzero to 0 */
      for (j = firstFixedNonzero+1; j < nvars; ++j)
      {
         /* fix variable */
         SCIP_CALL( inferVariableZero(scip, vars[j], cons, firstFixedNonzero, &infeasible, &tightened, &success) );
         assert( ! infeasible ); /* there should be no variables after firstFixedNonzero that are fixed to be nonzero */
         allVarFixed = allVarFixed && success;
         if ( tightened )
            ++(*ngen);
      }

      /* reset constraint age counter */
      if ( *ngen > ngenold )
      {
         SCIP_CALL( SCIPresetConsAge(scip, cons) );
      }

      /* delete constraint locally */
      if ( allVarFixed )
      {
         assert( !SCIPconsIsModifiable(cons) );
         SCIP_CALL( SCIPdelConsLocal(scip, cons) );
      }
   }

   return SCIP_OKAY;
}

/* ----------------------------- branching -------------------------------------*/

/** enforcement method
 *
 *  We check whether the current solution is feasible, i.e., contains at most one nonzero
 *  variable. If not, we branch along the lines indicated by Beale and Tomlin:
 *
 *  We first compute \f$W = \sum_{j=1}^n |x_i|\f$ and \f$w = \sum_{j=1}^n j\, |x_i|\f$. Then we
 *  search for the index \f$k\f$ that satisfies
 *  \f[
 *        k \leq \frac{w}{W} < k+1.
 *  \f]
 *  The branches are then
 *  \f[
 *        x_1 = 0, \ldots, x_k = 0 \qquad \mbox{and}\qquad x_{k+1} = 0, \ldots, x_n = 0.
 *  \f]
 *
 *  If the constraint contains two variables, the branching of course simplifies.
 *
 *  Depending on the parameters (@c branchnonzeros, @c branchweight) there are three ways to choose
 *  the branching constraint.
 *
 *  <TABLE>
 *  <TR><TD>@c branchnonzeros</TD><TD>@c branchweight</TD><TD>constraint chosen</TD></TR>
 *  <TR><TD>@c true          </TD><TD> ?             </TD><TD>most number of nonzeros</TD></TR>
 *  <TR><TD>@c false         </TD><TD> @c true       </TD><TD>maximal weight corresponding to nonzero variable</TD></TR>
 *  <TR><TD>@c false         </TD><TD> @c true       </TD><TD>largest sum of variable values</TD></TR>
 *  </TABLE>
 *
 *  @c branchnonzeros = @c false, @c branchweight = @c true allows the user to specify an order for
 *  the branching importance of the constraints (setting the weights accordingly).
 *
 *  Constraint branching can also be turned off using parameter @c branchsos.
 */
static
SCIP_RETCODE enforceSOS1(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   int                   nconss,             /**< number of constraints */
   SCIP_CONS**           conss,              /**< indicator constraints */
   SCIP_RESULT*          result              /**< result */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP_NODE* node1;
   SCIP_NODE* node2;
   SCIP_CONS* branchCons;
   SCIP_Real maxWeight;
   SCIP_VAR** vars;
   int nvars;
   int c;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( conss != NULL );
   assert( result != NULL );

   maxWeight = -SCIP_REAL_MAX;
   branchCons = NULL;

   SCIPdebugMessage("Enforcing SOS1 constraints <%s>.\n", SCIPconshdlrGetName(conshdlr) );
   *result = SCIP_FEASIBLE;

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );

   /* check each constraint */
   for (c = 0; c < nconss; ++c)
   {
      SCIP_CONS* cons;
      SCIP_Bool cutoff;
      SCIP_Real weight;
      int ngen;
      int cnt;
      int j;

      cons = conss[c];
      assert( cons != NULL );
      consdata = SCIPconsGetData(cons);
      assert( consdata != NULL );

      ngen = 0;
      cnt = 0;
      nvars = consdata->nvars;
      vars = consdata->vars;

      /* do nothing if there are not enough variables - this is usually eliminated by preprocessing */
      if ( nvars < 2 )
         continue;

      /* first perform propagation (it might happen that standard propagation is turned off) */
      SCIP_CALL( propSOS1(scip, cons, consdata, &cutoff, &ngen) );
      SCIPdebugMessage("propagating <%s> in enforcing (cutoff: %u, domain reductions: %d).\n", SCIPconsGetName(cons), cutoff, ngen);
      if ( cutoff )
      {
         *result = SCIP_CUTOFF;
         return SCIP_OKAY;
      }
      if ( ngen > 0 )
      {
         *result = SCIP_REDUCEDDOM;
         return SCIP_OKAY;
      }
      assert( ngen == 0 );

      /* check constraint */
      weight = 0.0;
      for (j = 0; j < nvars; ++j)
      {
         SCIP_Real val = REALABS(SCIPgetSolVal(scip, NULL, vars[j]));

         if ( ! SCIPisFeasZero(scip, val) )
         {
            if ( conshdlrdata->branchnonzeros )
               weight += 1.0;
            else
            {
               if ( conshdlrdata->branchweight )
               {
                  /* choose maximum nonzero-variable weight */
                  if ( consdata->weights[j] > weight )
                     weight = consdata->weights[j];
               }
               else
                  weight += val;
            }
            ++cnt;
         }
      }
      /* if constraint is violated */
      if ( cnt > 1 && weight > maxWeight )
      {
         maxWeight = weight;
         branchCons = cons;
      }
   }

   /* if all constraints are feasible */
   if ( branchCons == NULL )
   {
      SCIPdebugMessage("All SOS1 constraints are feasible.\n");
      return SCIP_OKAY;
   }

   /* if we should leave branching decision to branching rules */
   if ( ! conshdlrdata->branchsos )
   {
      *result = SCIP_INFEASIBLE;
      return SCIP_OKAY;
   }

   /* otherwise create branches */
   SCIPdebugMessage("Branching on constraint <%s> (weight: %f).\n", SCIPconsGetName(branchCons), maxWeight);
   consdata = SCIPconsGetData(branchCons);
   assert( consdata != NULL );
   nvars = consdata->nvars;
   vars = consdata->vars;

   if ( nvars == 2 )
   {
      SCIP_Bool infeasible;

      /* constraint is infeasible: */
      assert( ! SCIPisFeasZero(scip, SCIPgetSolVal(scip, NULL, vars[0])) && ! SCIPisFeasZero(scip, SCIPgetSolVal(scip, NULL, vars[1])) );

      /* create branches */
      SCIPdebugMessage("Creating two branches.\n");

      SCIP_CALL( SCIPcreateChild(scip, &node1, SCIPcalcNodeselPriority(scip, vars[0], SCIP_BRANCHDIR_DOWNWARDS, 0.0), SCIPcalcChildEstimate(scip, vars[0], 0.0) ) );
      SCIP_CALL( fixVariableZeroNode(scip, vars[0], node1, &infeasible) );
      assert( ! infeasible );

      SCIP_CALL( SCIPcreateChild(scip, &node2, SCIPcalcNodeselPriority(scip, vars[1], SCIP_BRANCHDIR_DOWNWARDS, 0.0), SCIPcalcChildEstimate(scip, vars[1], 0.0) ) );
      SCIP_CALL( fixVariableZeroNode(scip, vars[1], node2, &infeasible) );
      assert( ! infeasible );
   }
   else
   {
      SCIP_Bool infeasible;
      SCIP_Real weight1;
      SCIP_Real weight2;
      SCIP_Real nodeselest;
      SCIP_Real objest;
      SCIP_Real w;
      int j;
      int ind;
      int cnt;

      cnt = 0;

      weight1 = 0.0;
      weight2 = 0.0;

      /* compute weight */
      for (j = 0; j < nvars; ++j)
      {
         SCIP_Real val = REALABS(SCIPgetSolVal(scip, NULL, vars[j]));
         weight1 += val * (SCIP_Real) j;
         weight2 += val;

         if ( ! SCIPisFeasZero(scip, val) )
            ++cnt;
      }

      assert( cnt >= 2 );
      assert( !SCIPisFeasZero(scip, weight2) );
      w = weight1/weight2;  /*lint !e795*/

      ind = (int) SCIPfloor(scip, w);
      assert( 0 <= ind && ind < nvars-1 );

      /* branch on variable ind: either all variables up to ind or all variables after ind are zero */
      SCIPdebugMessage("Branching on variable <%s>.\n", SCIPvarGetName(vars[ind]));

      /* calculate node selection and objective estimate for node 1 */
      nodeselest = 0.0;
      objest = 0.0;
      for (j = 0; j <= ind; ++j)
      {
         nodeselest += SCIPcalcNodeselPriority(scip, vars[j], SCIP_BRANCHDIR_DOWNWARDS, 0.0);
         objest += SCIPcalcChildEstimate(scip, vars[j], 0.0);
      }
      /* take the average of the individual estimates */
      objest = objest/((SCIP_Real) ind + 1.0);

      /* create node 1 */
      SCIP_CALL( SCIPcreateChild(scip, &node1, nodeselest, objest) );
      for (j = 0; j <= ind; ++j)
      {
         SCIP_CALL( fixVariableZeroNode(scip, vars[j], node1, &infeasible) );
         assert( ! infeasible );
      }

      /* calculate node selection and objective estimate for node 1 */
      nodeselest = 0.0;
      objest = 0.0;
      for (j = ind+1; j < nvars; ++j)
      {
         nodeselest += SCIPcalcNodeselPriority(scip, vars[j], SCIP_BRANCHDIR_DOWNWARDS, 0.0);
         objest += SCIPcalcChildEstimate(scip, vars[j], 0.0);
      }
      /* take the average of the individual estimates */
      objest = objest/((SCIP_Real) (nvars - ind - 1));

      /* create node 2 */
      SCIP_CALL( SCIPcreateChild(scip, &node2, nodeselest, objest) );
      for (j = ind+1; j < nvars; ++j)
      {
         SCIP_CALL( fixVariableZeroNode(scip, vars[j], node2, &infeasible) );
         assert( ! infeasible );
      }
   }
   SCIP_CALL( SCIPresetConsAge(scip, branchCons) );
   *result = SCIP_BRANCHED;

   return SCIP_OKAY;
}


/* ----------------------------- separation ------------------------------------*/

/* initialitze tclique graph and create clique data */
static
SCIP_RETCODE initTCliquegraph(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONSHDLRDATA*    conshdlrdata,       /**< constraint handler data */
   SCIP_DIGRAPH*         conflictgraph,      /**< conflict graph */
   int                   nsos1vars,          /**< number of SOS1 variables */
   SCIP_SOL*             sol,                /**< LP solution to be separated (or NULL) */
   SCIP_Real             scaleval            /**< factor for scaling weights */
   )
{
   TCLIQUE_DATA* tcliquedata;
   int j;

   /* try to generate bound cuts */
   if ( ! tcliqueCreate(&conshdlrdata->tcliquegraph) )
      return SCIP_NOMEMORY;

   /* add nodes */
   for (j = 0; j < nsos1vars; ++j)
   {
      if ( ! tcliqueAddNode(conshdlrdata->tcliquegraph, j, 0 ) )
         return SCIP_NOMEMORY;
   }

   /* add edges */
   for (j = 0; j < nsos1vars; ++j)
   {
      int* succ;
      int nsucc;
      int succnode;
      int i;

      nsucc = SCIPdigraphGetNSuccessors(conflictgraph, j);
      succ = SCIPdigraphGetSuccessors(conflictgraph, j);

      for (i = 0; i < nsucc; ++i)
      {
         succnode = succ[i];

         if ( succnode > j && SCIPvarIsActive(nodeGetVarSOS1(conflictgraph, succnode)) )
         {
            if ( ! tcliqueAddEdge(conshdlrdata->tcliquegraph, j, succnode) )
               return SCIP_NOMEMORY;
         }
      }
   }
   if ( ! tcliqueFlush(conshdlrdata->tcliquegraph) )
      return SCIP_NOMEMORY;


   /* allocate clique data */
   SCIP_CALL( SCIPallocMemory(scip, &conshdlrdata->tcliquedata) );
   tcliquedata = conshdlrdata->tcliquedata;

   /* initialize clique data */
   tcliquedata->scip = scip;
   tcliquedata->sol = sol;
   tcliquedata->conshdlr = conshdlr;
   tcliquedata->conflictgraph = conflictgraph;
   tcliquedata->scaleval = scaleval;
   tcliquedata->ncuts = 0;
   tcliquedata->nboundcuts = conshdlrdata->nboundcuts;
   tcliquedata->strthenboundcuts = conshdlrdata->strthenboundcuts;
   tcliquedata->maxboundcuts = conshdlrdata->maxboundcutsroot;

   return SCIP_OKAY;
}


/* update weights of tclique graph */
static
SCIP_RETCODE updateWeightsTCliquegraph(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONSHDLRDATA*    conshdlrdata,       /**< constraint handler data */
   TCLIQUE_DATA*         tcliquedata,        /**< tclique data */
   SCIP_DIGRAPH*         conflictgraph,      /**< conflict graph */
   SCIP_SOL*             sol,                /**< LP solution to be separated (or NULL) */
   int                   nsos1vars           /**< number of SOS1 variables */
   )
{
   SCIP_Real scaleval;
   int j;

   scaleval = tcliquedata->scaleval;

   for (j = 0; j < nsos1vars; ++j)
   {
      SCIP_Real solval;
      SCIP_Real bound;
      SCIP_VAR* var;

      var = nodeGetVarSOS1(conflictgraph, j);
      solval = SCIPgetSolVal(scip, sol, var);

      if ( SCIPisFeasPositive(scip, solval) )
      {
         if ( conshdlrdata->strthenboundcuts )
            bound = REALABS( SCIPnodeGetSolvalVarboundUbSOS1(scip, conflictgraph, sol, j) );
         else
            bound = REALABS( SCIPvarGetUbLocal(var) );
      }
      else if ( SCIPisFeasNegative(scip, solval) )
      {
         if ( conshdlrdata->strthenboundcuts )
            bound = REALABS( SCIPnodeGetSolvalVarboundLbSOS1(scip, conflictgraph, sol, j) );
         else
            bound = REALABS( SCIPvarGetLbLocal(var) );
      }
      else
         bound = 0.0;

      solval = REALABS( solval );

      if ( ! SCIPisFeasZero(scip, bound) && ! SCIPisInfinity(scip, bound) )
      {
         SCIP_Real nodeweight = REALABS( solval/bound ) * scaleval;
         tcliqueChangeWeight(conshdlrdata->tcliquegraph, j, (int)nodeweight);
      }
      else
      {
         tcliqueChangeWeight(conshdlrdata->tcliquegraph, j, 0);
      }
   }

   return SCIP_OKAY;
}


/* adds bound cut(s) to separation storage */
static
SCIP_RETCODE addBoundCutSepa(
   SCIP*                 scip,               /**< SCIP pointer */
   TCLIQUE_DATA*         tcliquedata,        /**< clique data */
   SCIP_ROW*             rowlb,              /**< row for lower bounds (or NULL) */
   SCIP_ROW*             rowub,              /**< row for upper bounds (or NULL) */
   SCIP_Bool*            success             /**< pointer to store if bound cut was added */
   )
{
   assert( scip != NULL );
   assert( tcliquedata != NULL );
   assert( success != NULL);

   *success = FALSE;

   /* add cut for lower bounds */
   if ( rowlb != NULL )
   {
      if ( ! SCIProwIsInLP(rowlb) && SCIPisCutEfficacious(scip, NULL, rowlb) )
      {
         SCIP_Bool infeasible;

         SCIP_CALL( SCIPaddCut(scip, NULL, rowlb, FALSE, &infeasible) );
         assert( ! infeasible );
         SCIPdebug( SCIP_CALL( SCIPprintRow(scip, rowlb, NULL) ) );
         ++tcliquedata->nboundcuts;
         ++tcliquedata->ncuts;
         *success = TRUE;
      }
      SCIP_CALL( SCIPreleaseRow(scip, &rowlb) );
   }

   /* add cut for upper bounds */
   if ( rowub != NULL )
   {
      if ( ! SCIProwIsInLP(rowub) && SCIPisCutEfficacious(scip, NULL, rowub) )
      {
         SCIP_Bool infeasible;

         SCIP_CALL( SCIPaddCut(scip, NULL, rowub, FALSE, &infeasible) );
         assert( ! infeasible );
         SCIPdebug( SCIP_CALL( SCIPprintRow(scip, rowub, NULL) ) );
         ++tcliquedata->nboundcuts;
         ++tcliquedata->ncuts;
         *success = TRUE;
      }
      SCIP_CALL( SCIPreleaseRow(scip, &rowub) );
   }

   return SCIP_OKAY;
}


/** Generate bound constraint
 *
 *  We generate the row corresponding to the following simple valid inequalities:
 *  \f[
 *         \frac{x_1}{u_1} + \ldots + \frac{x_n}{u_n} \leq 1\qquad\mbox{and}\qquad
 *         \frac{x_1}{\ell_1} + \ldots + \frac{x_n}{\ell_1} \leq 1,
 *  \f]
 *  where \f$\ell_1, \ldots, \ell_n\f$ and \f$u_1, \ldots, u_n\f$ are the nonzero and finite lower and upper bounds of
 *  the variables \f$x_1, \ldots, x_n\f$. If an upper bound < 0 or a lower bound > 0, the constraint itself is
 *  redundant, so the cut is not applied (lower bounds > 0 and upper bounds < 0 are usually detected in presolving or
 *  propagation). Infinite bounds and zero are skipped. Thus \f$\ell_1, \ldots, \ell_n\f$ are all negative, which
 *  results in the \f$\leq\f$ inequality. In case of the presence of variable upper bounds, the bound inequality can
 *  be further strengthened.
 *
 *  Note that in fact, any mixture of nonzero finite lower and upper bounds would lead to a valid inequality as
 *  above. However, usually either the lower or upper bound is nonzero. Thus, the above inequalities are the most
 *  interesting.
 */
static
SCIP_RETCODE SCIPgenerateBoundInequalityFromSOS1Nodes(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DIGRAPH*         conflictgraph,      /**< conflict graph */
   int*                  nodes,              /**< conflict graph nodes for bound constraint */
   int                   nnodes,             /**< number of conflict graph nodes for bound constraint */
   SCIP_Real             rhs,                /**< right hand side of bound constraint */
   SCIP_Bool             local,              /**< in any case produce a local cut (even if local bounds of variables are valid globally) */
   SCIP_Bool             global,             /**< in any case produce a global cut */
   SCIP_Bool             strengthen,         /**< whether trying to strengthen bound constraint */
   SCIP_Bool             removable,          /**< should the inequality be removed from the LP due to aging or cleanup? */
   const char *          nameext,            /**< part of name of bound constraints */
   SCIP_ROW**            rowlb,              /**< output: row for lower bounds (or NULL if not needed) */
   SCIP_ROW**            rowub               /**< output: row for upper bounds (or NULL if not needed) */
   )
{
   char name[SCIP_MAXSTRLEN];
   SCIP_VAR* lbboundvar = NULL;
   SCIP_VAR* ubboundvar = NULL;
   SCIP_Bool locallbs;
   SCIP_Bool localubs;
   SCIP_VAR** vars;
   SCIP_Real* vals;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( conflictgraph != NULL );
   assert( ! local || ! global );
   assert( nodes != NULL );

   /* allocate buffer array */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, nnodes+1) );
   SCIP_CALL( SCIPallocBufferArray(scip, &vals, nnodes+1) );

   /* take care of upper bounds */
   if ( rowub != NULL )
   {
      SCIP_Bool useboundvar;
      int cnt;
      int j;

      /* loop through all variables. We check whether all bound variables (if existent) are equal; if this is the
       * case then the bound constraint can be strengthened */
      cnt = 0;
      localubs = local;
      useboundvar = strengthen;
      for (j = 0; j < nnodes; ++j)
      {
         SCIP_NODEDATA* nodedata;
         SCIP_VAR* var;
         SCIP_Real val;

         nodedata = (SCIP_NODEDATA*)SCIPdigraphGetNodeData(conflictgraph, nodes[j]);
         assert( nodedata != NULL );
         var = nodedata->var;
         assert( var != NULL );

         /* if variable is not involved in a variable bound constraint */
         if ( ! useboundvar || nodedata->ubboundvar == NULL )
         {
            useboundvar = FALSE;
            if ( localubs )
            {
               assert( ! global );
               val = SCIPvarGetUbLocal(var);
            }
            else
            {
               val = SCIPvarGetUbGlobal(var);
               if ( ! global && ! SCIPisFeasEQ(scip, val, SCIPvarGetUbLocal(var)) )
               {
                  localubs = TRUE;

                  /* restart 'for'-loop, since we need the local bounds of the variables */
                  j = -1;
                  cnt = 0;
                  continue;
               }
            }
         }
         else
         {
            /* in this case the cut is always valid globally */

            /* if we have a bound variable for the first time */
            if ( ubboundvar == NULL )
            {
               ubboundvar = nodedata->ubboundvar;
               val = nodedata->ubboundcoef;
            }
            /* else if the bound variable equals the stored bound variable */
            else if ( SCIPvarCompare(ubboundvar, nodedata->ubboundvar) == 0 )
            {
               val = nodedata->ubboundcoef;
            }
            else /* else use bounds on the variables */
            {
               useboundvar = FALSE;

               /* restart 'for'-loop */
               j = -1;
               cnt = 0;
               continue;
            }
         }

         /* should not apply the cut if a variable is fixed to be negative -> constraint is redundant */
         if ( SCIPisNegative(scip, val) )
            break;

         /* store variable if relevant for bound inequality */
         if ( ! SCIPisInfinity(scip, val) && ! SCIPisZero(scip, val) )
         {
            vars[cnt] = var;
            vals[cnt++] = 1.0/val;
         }
      }

      /* if cut is meaningful */
      if ( j == nnodes && cnt >= 2 )
      {
         if ( useboundvar )
         {
            /* add bound variable to array */
            vars[cnt] = ubboundvar;
            vals[cnt++] = -rhs;
            assert(ubboundvar != NULL );

            /* create upper bound inequality if at least two of the bounds are finite and nonzero */
            (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "sosub#%s", nameext);
            SCIP_CALL( SCIPcreateEmptyRowCons(scip, rowub, conshdlr, name, -SCIPinfinity(scip), 0.0, localubs, FALSE, removable) );
            SCIP_CALL( SCIPaddVarsToRow(scip, *rowub, cnt, vars, vals) );
            SCIPdebug( SCIP_CALL( SCIPprintRow(scip, *rowub, NULL) ) );
         }
         else
         {
            /* create upper bound inequality if at least two of the bounds are finite and nonzero */
            (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "sosub#%s", nameext);
            SCIP_CALL( SCIPcreateEmptyRowCons(scip, rowub, conshdlr, name, -SCIPinfinity(scip), rhs, localubs, FALSE, removable) );
            SCIP_CALL( SCIPaddVarsToRow(scip, *rowub, cnt, vars, vals) );
            SCIPdebug( SCIP_CALL( SCIPprintRow(scip, *rowub, NULL) ) );
         }
      }
   }


   /* take care of lower bounds */
   if ( rowlb != NULL )
   {
      SCIP_Bool useboundvar;
      int cnt;
      int j;

      /* loop through all variables. We check whether all bound variables (if existent) are equal; if this is the
       * case then the bound constraint can be strengthened */
      cnt = 0;
      locallbs = local;
      useboundvar = strengthen;
      for (j = 0; j < nnodes; ++j)
      {
         SCIP_NODEDATA* nodedata;
         SCIP_VAR* var;
         SCIP_Real val;

         nodedata = (SCIP_NODEDATA*)SCIPdigraphGetNodeData(conflictgraph, nodes[j]);
         assert( nodedata != NULL );
         var = nodedata->var;
         assert( var != NULL );

         /* if variable is not involved in a variable bound constraint */
         if ( ! useboundvar || nodedata->lbboundvar == NULL )
         {
            useboundvar = FALSE;
            if ( locallbs )
            {
               assert( ! global );
               val = SCIPvarGetLbLocal(var);
            }
            else
            {
               val = SCIPvarGetLbGlobal(var);
               if ( ! global && ! SCIPisFeasEQ(scip, val, SCIPvarGetLbLocal(var)) )
               {
                  locallbs = TRUE;

                  /* restart 'for'-loop, since we need the local bounds of the variables */
                  j = -1;
                  cnt = 0;
                  continue;
               }
            }
         }
         else
         {
            /* in this case the cut is always valid globally */

            /* if we have a bound variable for the first time */
            if ( lbboundvar == NULL )
            {
               lbboundvar = nodedata->lbboundvar;
               val = nodedata->lbboundcoef;
            }
            /* else if the bound variable equals the stored bound variable */
            else if ( SCIPvarCompare(lbboundvar, nodedata->lbboundvar) == 0 )
            {
               val = nodedata->lbboundcoef;
            }
            else /* else use bounds on the variables */
            {
               useboundvar = FALSE;

               /* restart 'for'-loop */
               j = -1;
               cnt = 0;
               continue;
            }
         }

         /* should not apply the cut if a variable is fixed to be positive -> constraint is redundant */
         if ( SCIPisPositive(scip, val) )
            break;

         /* store variable if relevant for bound inequality */
         if ( ! SCIPisInfinity(scip, val) && ! SCIPisZero(scip, val) )
         {
            vars[cnt] = var;
            vals[cnt++] = 1.0/val;
         }
      }

      /* if cut is meaningful */
      if ( j == nnodes && cnt >= 2 )
      {
         if ( useboundvar )
         {
            /* add bound variable to array */
            vars[cnt] = lbboundvar;
            vals[cnt++] = -rhs;
            assert(lbboundvar != NULL );

            /* create upper bound inequality if at least two of the bounds are finite and nonzero */
            (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "soslb#%s", nameext);
            SCIP_CALL( SCIPcreateEmptyRowCons(scip, rowlb, conshdlr, name, -SCIPinfinity(scip), 0.0, locallbs, FALSE, TRUE) );
            SCIP_CALL( SCIPaddVarsToRow(scip, *rowlb, cnt, vars, vals) );
            SCIPdebug( SCIP_CALL( SCIPprintRow(scip, *rowlb, NULL) ) );
         }
         else
         {
            /* create upper bound inequality if at least two of the bounds are finite and nonzero */
            (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "soslb#%s", nameext);
            SCIP_CALL( SCIPcreateEmptyRowCons(scip, rowlb, conshdlr, name, -SCIPinfinity(scip), rhs, locallbs, FALSE, TRUE) );
            SCIP_CALL( SCIPaddVarsToRow(scip, *rowlb, cnt, vars, vals) );
            SCIPdebug( SCIP_CALL( SCIPprintRow(scip, *rowlb, NULL) ) );
         }
      }
   }

   /* free buffer array */
   SCIPfreeBufferArray(scip, &vals);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}


/** generates bound cuts using a clique found by algorithm for maximum weight clique
 *  and decides whether to stop generating cliques with the algorithm for maximum weight clique
 */
static
TCLIQUE_NEWSOL(tcliqueNewsolClique)
{
   TCLIQUE_WEIGHT minweightinc;

   assert( acceptsol != NULL );
   assert( stopsolving != NULL );
   assert( tcliquedata != NULL );

   /* we don't accept the solution as new incumbent, because we want to find many violated clique inequalities */
   *acceptsol = FALSE;
   *stopsolving = FALSE;

   /* slightly increase the minimal weight for additional cliques */
   minweightinc = (cliqueweight - *minweight)/10;
   minweightinc = MAX(minweightinc, 1);
   *minweight += minweightinc;

   /* adds cut if weight of the clique is greater than 1 */
   if( cliqueweight > tcliquedata->scaleval )
   {
      SCIP* scip;
      SCIP_SOL* sol;
      SCIP_Real unscaledweight;
      SCIP_Real solval;
      SCIP_Real bound;
      SCIP_VAR* var;
      int node;
      int i;

      scip = tcliquedata->scip;
      sol = tcliquedata->sol;
      assert( scip != NULL );

      /* calculate the weight of the clique in unscaled fractional variable space */
      unscaledweight = 0.0;
      for( i = 0; i < ncliquenodes; i++ )
      {
         node = cliquenodes[i];
         var = nodeGetVarSOS1(tcliquedata->conflictgraph, node);
         solval = SCIPgetSolVal(scip, sol, var);

         if ( SCIPisFeasPositive(scip, solval) )
         {
            if ( tcliquedata->strthenboundcuts )
               bound = REALABS( SCIPnodeGetSolvalVarboundUbSOS1(scip, tcliquedata->conflictgraph, sol, node) );
            else
               bound = REALABS( SCIPvarGetUbLocal(var) );
         }
         else if ( SCIPisFeasNegative(scip, solval) )
         {
            if ( tcliquedata->strthenboundcuts )
               bound = REALABS( SCIPnodeGetSolvalVarboundLbSOS1(scip, tcliquedata->conflictgraph, sol, node) );
            else
               bound = REALABS( SCIPvarGetLbLocal(var) );
         }
         else
            bound = 0.0;

         solval = REALABS( solval );

         if ( ! SCIPisFeasZero(scip, bound) && ! SCIPisInfinity(scip, bound) )
            unscaledweight += REALABS( solval/bound );
      }

      if( SCIPisEfficacious(scip, unscaledweight - 1.0) )
      {
         char nameext[SCIP_MAXSTRLEN];
         SCIP_ROW* rowlb = NULL;
         SCIP_ROW* rowub = NULL;
         SCIP_Bool success;

         /* generate bound inequalities for lower and upper bound case
          * NOTE: tests have shown that non-removable rows give the best results */
         (void) SCIPsnprintf(nameext, SCIP_MAXSTRLEN, "%d", tcliquedata->nboundcuts);
         if( SCIPgenerateBoundInequalityFromSOS1Nodes(scip, tcliquedata->conshdlr, tcliquedata->conflictgraph,
               cliquenodes, ncliquenodes, 1.0, FALSE, FALSE, tcliquedata->strthenboundcuts, FALSE, nameext, &rowlb, &rowub) != SCIP_OKAY )
         {
            SCIPerrorMessage("unexpected error in bound cut creation.\n");
            SCIPABORT();
         }

         /* add bound cut(s) to separation storage if existent */
         if ( addBoundCutSepa(scip, tcliquedata, rowlb, rowub, &success) != SCIP_OKAY )
         {
            SCIPerrorMessage("unexpected error in bound cut creation.\n");
            SCIPABORT();
         }

         /* if at least one cut has been added */
         if ( success )
         {
            SCIPdebugMessage(" -> found bound cut corresponding to clique (act=%g)\n", unscaledweight);

            /* if we found more than half the cuts we are allowed to generate, we accept the clique as new incumbent,
             * such that only more violated cuts are generated afterwards
             */
            if( tcliquedata->maxboundcuts >= 0 )
            {
               if( tcliquedata->ncuts > tcliquedata->maxboundcuts/2 )
                  *acceptsol = TRUE;
               if( tcliquedata->ncuts >= tcliquedata->maxboundcuts )
                  *stopsolving = TRUE;
            }
         }
         else
            *stopsolving = TRUE;
      }
   }
}


/** separate bound inequalities from conflict graph */
static
SCIP_RETCODE sepaBoundInequalitiesFromGraph(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONSHDLRDATA*    conshdlrdata,       /**< constraint handler data */
   SCIP_SOL*             sol,                /**< LP solution to be separated (or NULL) */
   int                   maxboundcuts,       /**< maximal number of bound cuts separated per separation round (-1: no limit) */
   int*                  ngen,               /**< pointer to store number of cuts generated */
   SCIP_RESULT*          result              /**< pointer to store result of separation */
   )
{
   SCIP_DIGRAPH* conflictgraph;
   TCLIQUE_DATA* tcliquedata;
   TCLIQUE_WEIGHT cliqueweight;
   TCLIQUE_STATUS tcliquestatus;
   int nsos1vars;

   SCIP_Real scaleval = 1000.0;                  /* factor for scaling weights */
   int maxtreenodes = 10000;                     /* maximal number of nodes of b&b tree */
   int maxzeroextensions = 1000;                 /* maximal number of zero-valued variables extending the clique (-1: no limit) */
   int backtrackfreq = 1000;                     /* frequency for premature backtracking up to tree level 1 (0: no backtracking) */
   int ntreenodes;
   int* cliquenodes;
   int ncliquenodes;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( conshdlrdata != NULL );
   assert( ngen != NULL );
   assert( result != NULL );

   /* get conflict graph */
   conflictgraph = SCIPgetConflictgraphSOS1(conshdlr);
   assert( conflictgraph != NULL );

   /* get number of SOS1 variables */
   nsos1vars = SCIPgetNSOS1Vars(conshdlr);

   /* initialize tclique graph if not done already */
   if ( conshdlrdata->tcliquegraph == NULL )
   {
      SCIP_CALL( initTCliquegraph(scip, conshdlr, conshdlrdata, conflictgraph, nsos1vars, sol, scaleval) );
   }
   tcliquedata = conshdlrdata->tcliquedata;
   tcliquedata->maxboundcuts = maxboundcuts;
   tcliquedata->ncuts = 0;

   /* update the weights of the tclique graph */
   SCIP_CALL( updateWeightsTCliquegraph(scip, conshdlrdata, tcliquedata, conflictgraph, sol, nsos1vars) );

   /* allocate buffer array */
   SCIP_CALL( SCIPallocBufferArray(scip, &cliquenodes, nsos1vars) );

   /* start algorithm to find maximum weight cliques and use them to generate bound cuts */
   tcliqueMaxClique(tcliqueGetNNodes, tcliqueGetWeights, tcliqueIsEdge, tcliqueSelectAdjnodes,
      conshdlrdata->tcliquegraph, tcliqueNewsolClique, tcliquedata,
      cliquenodes, &ncliquenodes, &cliqueweight, (int)scaleval-1, (int)scaleval+1,
      maxtreenodes, backtrackfreq, maxzeroextensions, -1, &ntreenodes, &tcliquestatus);

   /* free buffer array */
   SCIPfreeBufferArray(scip, &cliquenodes);

   /* get number of cuts of current separation round */
   *ngen = tcliquedata->ncuts;

   /* update number of bound cuts in separator data */
   conshdlrdata->nboundcuts = tcliquedata->nboundcuts;

   /* evaluate the result of the separation */
   if ( *ngen > 0 )
      *result = SCIP_SEPARATED;
   else
      *result = SCIP_DIDNOTFIND;

   return SCIP_OKAY;
}


/** Generate a bound constraint from the variables of an SOS1 constraint (see SCIPgenerateBoundInequalityFromSOS1Nodes() for more information) */
static
SCIP_RETCODE SCIPgenerateBoundInequalityFromSOS1Cons(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONS*            cons,               /**< SOS1 constraint */
   SCIP_Bool             local,              /**< in any case produce a local cut (even if local bounds of variables are valid globally) */
   SCIP_Bool             global,             /**< in any case produce a global cut */
   SCIP_Bool             strengthen,         /**< whether trying to strengthen bound constraint */
   SCIP_Bool             removable,          /**< should the inequality be removed from the LP due to aging or cleanup? */
   SCIP_ROW**            rowlb,              /**< output: row for lower bounds (or NULL if not needed) */
   SCIP_ROW**            rowub               /**< output: row for upper bounds (or NULL if not needed) */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   int* nodes;
   int nvars;
   int j;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( cons != NULL );

   /* get constraint data */
   consdata = SCIPconsGetData(cons);
   assert( consdata != NULL );
   assert( consdata->vars != NULL );
   nvars = consdata->nvars;

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );

   /* allocate buffer array */
   SCIP_CALL( SCIPallocBufferArray(scip, &nodes, nvars) );

   /* get nodes in the conflict graph */
   for (j = 0; j < nvars; ++j)
   {
      nodes[j] = varGetNodeSOS1(conshdlr, consdata->vars[j]);
      assert( nodes[j] >= 0 );
   }

   /* generate bound constraint from conflict graph nodes */
   SCIP_CALL( SCIPgenerateBoundInequalityFromSOS1Nodes(scip, conshdlr, conshdlrdata->conflictgraph, nodes, nvars, 1.0, local, global, strengthen, removable, SCIPconsGetName(cons), rowlb, rowub) );

   /* free buffer array */
   SCIPfreeBufferArray(scip, &nodes);

   return SCIP_OKAY;
}


/** initialize or separate bound inequalities from SOS1 constraints */
static
SCIP_RETCODE initsepaBoundInequalityFromSOS1Cons(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONSHDLRDATA*    conshdlrdata,       /**< constraint handler data */
   SCIP_CONS**           conss,              /**< SOS1 constraints */
   int                   nconss,             /**< number of SOS1 constraints */
   SCIP_SOL*             sol,                /**< LP solution to be separated (or NULL) */
   SCIP_Bool             solvedinitlp,       /**< TRUE if initial LP relaxation at a node is solved */
   int                   maxboundcuts,       /**< maximal number of bound cuts separated per separation round (-1: no limit) */
   int*                  ngen,               /**< pointer to store number of cuts generated (or NULL) */
   SCIP_RESULT*          result              /**< pointer to store result of separation (or NULL) */
   )
{
   SCIP_Bool cutoff = FALSE;
   int c;

   assert( scip != NULL );
   assert( conshdlrdata != NULL );
   assert( conss != NULL );

   if ( result != NULL )
      *result = SCIP_DIDNOTFIND;

   for (c = 0; c < nconss; ++c)
   {
      SCIP_CONSDATA* consdata;
      SCIP_ROW* row;

      assert( conss != NULL );
      assert( conss[c] != NULL );
      consdata = SCIPconsGetData(conss[c]);
      assert( consdata != NULL );

      if ( solvedinitlp )
         SCIPdebugMessage("Separating inequalities for SOS1 constraint <%s>.\n", SCIPconsGetName(conss[c]) );
      else
         SCIPdebugMessage("Checking for initial rows for SOS1 constraint <%s>.\n", SCIPconsGetName(conss[c]) );

      /* possibly generate rows if not yet done */
      if ( consdata->rowub == NULL || consdata->rowlb == NULL )
      {
         SCIP_ROW* rowlb = NULL;
         SCIP_ROW* rowub = NULL;

         SCIP_CALL( SCIPgenerateBoundInequalityFromSOS1Cons(scip, conshdlr, conss[c], FALSE, TRUE, TRUE, FALSE, &rowlb, &rowub) );

         /* if row(s) should be globally stored in constraint data */
         if ( rowlb != NULL )
         {
            consdata->rowlb = rowlb;
         }
         if ( rowub != NULL )
         {
            consdata->rowub = rowub;
         }
      }

      /* put corresponding rows into LP */
      row = consdata->rowub;
      if ( row != NULL && ! SCIProwIsInLP(row) && ( solvedinitlp || SCIPisCutEfficacious(scip, sol, row) ) )
      {
         assert( SCIPisInfinity(scip, -SCIProwGetLhs(row)) && ( SCIPisEQ(scip, SCIProwGetRhs(row), 1.0) || SCIPisEQ(scip, SCIProwGetRhs(row), 0.0) ) );

         SCIP_CALL( SCIPaddCut(scip, NULL, row, FALSE, &cutoff) );
         if ( cutoff && result != NULL )
            break;
         assert( ! cutoff );
         SCIPdebug( SCIP_CALL( SCIPprintRow(scip, row, NULL) ) );

         if ( solvedinitlp )
         {
            assert( ngen != NULL );
            SCIP_CALL( SCIPresetConsAge(scip, conss[c]) );
            ++(*ngen);
         }
      }
      row = consdata->rowlb;
      if ( row != NULL && ! SCIProwIsInLP(row) && ( solvedinitlp || SCIPisCutEfficacious(scip, sol, row) ) )
      {
         assert( SCIPisInfinity(scip, -SCIProwGetLhs(row)) && ( SCIPisEQ(scip, SCIProwGetRhs(row), 1.0) || SCIPisEQ(scip, SCIProwGetRhs(row), 0.0) ) );

         SCIP_CALL( SCIPaddCut(scip, NULL, row, FALSE, &cutoff) );
         if ( cutoff && result != NULL )
            break;
         assert( ! cutoff );
         SCIPdebug( SCIP_CALL( SCIPprintRow(scip, row, NULL) ) );

         if ( solvedinitlp )
         {
            assert( ngen != NULL );
            SCIP_CALL( SCIPresetConsAge(scip, conss[c]) );
            ++(*ngen);
         }
      }

      if ( ngen != NULL && maxboundcuts >= 0 && *ngen >= maxboundcuts )
         break;
   }

   if ( cutoff )
      *result = SCIP_CUTOFF;
   else if ( ngen != NULL && *ngen > 0 )
      *result = SCIP_SEPARATED;

   return SCIP_OKAY;
}


/** check whether var1 is a bound variable of var0; i.e., var0 >= c * var1 or var0 <= d * var1.
 *  If true, then add this information to the node data of the conflict graph.
 */
static
SCIP_RETCODE detectVarboundSOS1(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONSHDLR*        conshdlr,           /**< SOS1 constraint handler */
   SCIP_CONSHDLRDATA*    conshdlrdata,       /**< SOS1 constraint handler data */
   SCIP_VAR*             var0,               /**< first variable */
   SCIP_VAR*             var1,               /**< second variable */
   SCIP_Real             val0,               /**< first coefficient */
   SCIP_Real             val1                /**< second coefficient */
   )
{
   int node0;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( conshdlrdata != NULL );
   assert( var0 != NULL && var1 != NULL );

   /* get nodes of variable in the conflict graph (node = -1 if no SOS1 variable) */
   node0 = varGetNodeSOS1(conshdlr, var0);

   /* if var0 is an SOS1 variable */
   if ( node0 >= 0 )
   {
      SCIP_Real val;

      assert( ! SCIPisFeasZero(scip, val0) );
      val = -val1/val0;

      /* check variable bound relation of variables */

      /* handle lower bound case */
      if ( SCIPisFeasNegative(scip, val0) && SCIPisFeasNegative(scip, val) )
      {
         SCIP_NODEDATA* nodedata;

         /* get node data of the conflict graph */
         nodedata = (SCIP_NODEDATA*)SCIPdigraphGetNodeData(conshdlrdata->conflictgraph, node0);

         /* @todo: maybe save multiple variable bounds for each SOS1 variable */
         if ( nodedata->lbboundvar == NULL )
         {
            /* add variable bound information to node data */
            nodedata->lbboundvar = var1;
            nodedata->lbboundcoef = val;

            SCIPdebugMessage("detected variable bound constraint %s >= %f %s.\n", SCIPvarGetName(var0), val, SCIPvarGetName(var1));
         }
      }
      /* handle upper bound case */
      else if ( SCIPisFeasPositive(scip, val0) && SCIPisFeasPositive(scip, val) )
      {
         SCIP_NODEDATA* nodedata;
         assert( SCIPisFeasPositive(scip, val0) );

         /* get node data of the conflict graph */
         nodedata = (SCIP_NODEDATA*)SCIPdigraphGetNodeData(conshdlrdata->conflictgraph, node0);

         if ( nodedata->ubboundvar == NULL )
         {
            /* add variable bound information to node data */
            nodedata->ubboundvar = var1;
            nodedata->ubboundcoef = val;

            SCIPdebugMessage("detected variable bound constraint %s <= %f %s.\n", SCIPvarGetName(var0), val, SCIPvarGetName(var1));
         }
      }
   }

   return SCIP_OKAY;
}


/* pass connected component @p C of the conflict graph and check whether all the variables correspond to a unique variable upper bound variable @p z,
 *  i.e., \f$x_i \leq u_i z\f$ for every \f$i\in C\f$.
 *
 *  Note: if the upper bound variable is not unique, then bound inequalities usually cannot be strengthened.
 */
static
SCIP_RETCODE passConComponentVarbound(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_DIGRAPH*         conflictgraph,      /**< conflict graph */
   int                   node,               /**< current node of connected component */
   SCIP_VAR*             boundvar,           /**< bound variable of connected component */
   SCIP_Bool             checklb,            /**< whether to check lower bound variable (else upper bound variable) */
   SCIP_Bool*            processed,          /**< states for each variable whether it has been processed */
   int*                  concomp,            /**< current connected component */
   int*                  nconcomp,           /**< pointer to store number of elements of connected component */
   SCIP_Bool*            unique              /**< pointer to store whether bound variable is unique */
   )
{
   int* succ;
   int nsucc;
   int s;

   assert( scip != NULL );
   assert( conflictgraph != NULL );
   assert( processed != NULL );
   assert( concomp != NULL );
   assert( nconcomp != NULL );
   assert( unique != NULL );

   processed[node] = TRUE;
   concomp[(*nconcomp)++] = node;

   /* if bound variable of connected component without new node is unique */
   if ( unique )
   {
      SCIP_NODEDATA* nodedata;
      SCIP_VAR* comparevar;
      nodedata = (SCIP_NODEDATA*)SCIPdigraphGetNodeData(conflictgraph, node);
      assert( nodedata != NULL );

      if ( checklb )
         comparevar = nodedata->lbboundvar;
      else
         comparevar = nodedata->ubboundvar;

      /* check whether bound variable is unique for connected component without new node */
      if ( boundvar == NULL )
      {
         if ( comparevar != NULL )
            unique = FALSE;
      }
      else
      {
         if ( comparevar == NULL )
            unique = FALSE;
         else if ( SCIPvarCompare(boundvar, comparevar) != 0 )
            unique = FALSE;
      }
   }

   /* pass through successor variables */
   nsucc = SCIPdigraphGetNSuccessors(conflictgraph, node);
   succ = SCIPdigraphGetSuccessors(conflictgraph, node);
   for (s = 0; s < nsucc; ++s)
   {
      if ( ! processed[succ[s]] )
         SCIP_CALL( passConComponentVarbound(scip, conflictgraph, succ[s], boundvar, checklb, processed, concomp, nconcomp, unique) );
   }

   return SCIP_OKAY;
}


/** for each connected component @p C of the conflict graph check whether all the variables correspond to a unique variable upper bound variable @p z
 *  (e.g., for the upper bound case this means that \f$x_i \leq u_i z\f$ for every \f$i\in C\f$).
 *
 *  Note: if the bound variable is not unique, then bound inequalities usually cannot be strengthened.
 */
static
SCIP_RETCODE checkConComponentsVarbound(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_DIGRAPH*         conflictgraph,      /**< conflict graph */
   int                   nsos1vars,          /**< number of SOS1 variables */
   SCIP_Bool             checklb             /**< whether to check lower bound variable (else check upper bound variable) */
   )
{
   SCIP_Bool* processed;  /* states for each variable whether it has been processed */
   int* concomp;          /* current connected component */
   int nconcomp;
   int j;

   assert( scip != NULL );
   assert( conflictgraph != NULL );

   /* allocate buffer arrays and initialize 'processed' array */
   SCIP_CALL( SCIPallocBufferArray(scip, &processed, nsos1vars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &concomp, nsos1vars) );
   for (j = 0; j < nsos1vars; ++j)
      processed[j] = FALSE;

   /* run through all SOS1 variables */
   for (j = 0; j < nsos1vars; ++j)
   {
      /* if variable belongs to a connected component that has not been processed so far */
      if ( ! processed[j] )
      {
         SCIP_NODEDATA* nodedata;
         SCIP_VAR* boundvar;
         SCIP_Bool unique;
         int* succ;
         int nsucc;
         int s;

         nodedata = (SCIP_NODEDATA*)SCIPdigraphGetNodeData(conflictgraph, j);
         assert( nodedata != NULL );

         if ( checklb )
            boundvar = nodedata->lbboundvar;
         else
            boundvar = nodedata->ubboundvar;
         unique = TRUE;

         processed[j] = TRUE;
         concomp[0] = j;
         nconcomp = 1;

         /* pass through successor variables */
         nsucc = SCIPdigraphGetNSuccessors(conflictgraph, j);
         succ = SCIPdigraphGetSuccessors(conflictgraph, j);
         for (s = 0; s < nsucc; ++s)
         {
            if ( ! processed[succ[s]] )
               SCIP_CALL( passConComponentVarbound(scip, conflictgraph, succ[s], boundvar, checklb, processed, concomp, &nconcomp, &unique) );
         }

         /* if the connected component has a unique bound variable */
         if ( unique && boundvar != NULL )
         {
            for (s = 0; s < nconcomp; ++s)
            {
               nodedata = (SCIP_NODEDATA*)SCIPdigraphGetNodeData(conflictgraph, concomp[s]);
               assert( processed[concomp[s]] == TRUE );
               assert( nodedata != NULL );

               if ( checklb )
                  nodedata->lbboundcomp = TRUE;
               else
                  nodedata->ubboundcomp = TRUE;
            }
            SCIPdebugMessage("Found a connected component of size <%i> with unique bound variable.\n", nconcomp);
         }
      }
   }

   /* free buffer arrays */
   SCIPfreeBufferArray(scip, &concomp);
   SCIPfreeBufferArray(scip, &processed);

   return SCIP_OKAY;
}


/** check all linear constraints for variable bound constraints of the form c*z <= x <= d*z, where @p x is some SOS1
 *  variable and @p z some arbitrary variable (not necessarily binary)
 */
static
SCIP_RETCODE checkLinearConssVarboundSOS1(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONSHDLR*        conshdlr,           /**< SOS1 constraint handler */
   SCIP_CONSHDLRDATA*    conshdlrdata,       /**< SOS1 constraint handler data */
   SCIP_CONS**           linconss,           /**< linear constraints */
   int                   nlinconss           /**< number of linear constraints */
   )
{
   int c;

   /* loop through linear constraints */
   for (c = 0; c < nlinconss; ++c)
   {
      SCIP_CONS* lincons;
      int nvars;

      lincons = linconss[c];

      /* variable bound constraints only contain two variables */
      nvars = SCIPgetNVarsLinear(scip, lincons);
      if ( nvars == 2 )
      {
         SCIP_VAR** vars;
         SCIP_Real* vals;
         SCIP_VAR* var0;
         SCIP_VAR* var1;
         SCIP_Real lhs;
         SCIP_Real rhs;

         /* get constraint data */
         vars = SCIPgetVarsLinear(scip, lincons);
         vals = SCIPgetValsLinear(scip, lincons);
         lhs = SCIPgetLhsLinear(scip, lincons);
         rhs = SCIPgetRhsLinear(scip, lincons);

         var0 = vars[0];
         var1 = vars[1];
         assert( var0 != NULL && var1 != NULL );

         /* at least one variable should be an SOS1 variable */
         if ( varIsSOS1(conshdlr, var0) || varIsSOS1(conshdlr, var1) )
         {
            SCIP_Real val0;
            SCIP_Real val1;

            /* check whether right hand side or left hand side of constraint is zero */
            if ( SCIPisFeasZero(scip, lhs) )
            {
               val0 = -vals[0];
               val1 = -vals[1];

               /* check whether the two variables are in a variable bound relation */
               SCIP_CALL( detectVarboundSOS1(scip, conshdlr, conshdlrdata, var0, var1, val0, val1) );
               SCIP_CALL( detectVarboundSOS1(scip, conshdlr, conshdlrdata, var1, var0, val1, val0) );
            }
            else if( SCIPisFeasZero(scip, rhs) )
            {
               val0 = vals[0];
               val1 = vals[1];

               /* check whether the two variables are in a variable bound relation */
               SCIP_CALL( detectVarboundSOS1(scip, conshdlr, conshdlrdata, var0, var1, val0, val1) );
               SCIP_CALL( detectVarboundSOS1(scip, conshdlr, conshdlrdata, var1, var0, val1, val0) );
            }
         }
      }
   }

   return SCIP_OKAY;
}


/** set node data of conflict graph nodes */
static
SCIP_RETCODE setNodeDataSOS1(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONSHDLRDATA*    conshdlrdata,       /**< SOS1 constraint handler data */
   int                   nsos1conss,         /**< number of SOS1 constraints */
   int                   nsos1vars           /**< number of SOS1 variables */
   )
{
   SCIP_CONSHDLR* linconshdlr;
   SCIP_CONS** linconss;
   int nlinconss;

   /* if no SOS1 variables exist -> exit */
   if ( nsos1vars == 0 )
      return SCIP_OKAY;

   /* get constraint handler data of linear constraints */
   linconshdlr = SCIPfindConshdlr(scip, "linear");
   if ( linconshdlr == NULL )
      return SCIP_OKAY;

   /* get linear constraints and number of linear constraints */
   nlinconss = SCIPconshdlrGetNConss(linconshdlr);
   linconss = SCIPconshdlrGetConss(linconshdlr);

   /* check linear constraints for variable bound constraints */
   SCIP_CALL( checkLinearConssVarboundSOS1(scip, conshdlr, conshdlrdata, linconss, nlinconss) );

   /* for each connected component of the conflict graph check whether all the variables correspond to a unique variable
    * upper bound variable */
   SCIP_CALL( checkConComponentsVarbound(scip, conshdlrdata->conflictgraph, conshdlrdata->nsos1vars, TRUE) );
   SCIP_CALL( checkConComponentsVarbound(scip, conshdlrdata->conflictgraph, conshdlrdata->nsos1vars, FALSE) );

   return SCIP_OKAY;
}


/* initialize conflictgraph and create hashmap for SOS1 variables */
static
SCIP_RETCODE initConflictgraph(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_CONSHDLRDATA*    conshdlrdata,       /**< constraint handler data */
   SCIP_CONS**           conss,              /**< SOS1 constraints */
   int                   nconss              /**< number of SOS1 constraints */
   )
{
   SCIP_Bool* nodecreated; /* nodecreated[i] = TRUE if a node in the conflictgraph is already created for index i
                            * (with i index of the original variables) */
   int* nodeorig;          /* nodeorig[i] = node of original variable x_i in the conflictgraph */
   int ntotalvars;
   int cntsos;
   int i;
   int j;
   int c;

   assert( conshdlrdata != NULL );
   assert( nconss == 0 || conss != NULL );

   /* get the number of original problem variables */
   ntotalvars = SCIPgetNTotalVars(scip);

   /* initialize vector 'nodecreated' */
   SCIP_CALL( SCIPallocBufferArray(scip, &nodeorig, ntotalvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nodecreated, ntotalvars) );
   for (i = 0; i < ntotalvars; ++i)
      nodecreated[i] = FALSE;

   /* compute number of SOS1 variables */
   cntsos = 0;
   for (c = 0; c < nconss; ++c)
   {
      SCIP_CONSDATA* consdata;
      SCIP_VAR** vars;
      int nvars;

      assert( conss[c] != NULL );

      /* get constraint data field of the constraint */
      consdata = SCIPconsGetData(conss[c]);
      assert( consdata != NULL );

      /* get variables and number of variables of constraint */
      nvars = consdata->nvars;
      vars = consdata->vars;

      /* update number of SOS1 variables */
      for (i = 0; i < nvars; ++i)
      {
         SCIP_VAR* var;

         var = vars[i];

         if ( SCIPvarGetStatus(var) != SCIP_VARSTATUS_FIXED )
         {
            int ind;

            ind = SCIPvarGetIndex(var);
            assert( ind >= 0 && ind < ntotalvars );
            if ( ! nodecreated[ind] )
            {
               nodecreated[ind] = TRUE; /* mark node as counted */
               nodeorig[ind] = cntsos;
               ++cntsos;
            }
         }
      }
   }
   if ( cntsos <= 0 )
   {
      /* free buffer arrays */
      SCIPfreeBufferArray(scip, &nodecreated);
      SCIPfreeBufferArray(scip, &nodeorig);
      conshdlrdata->nsos1vars = 0;
      return SCIP_OKAY;
   }

   /* reinitialize vector 'nodecreated' */
   for (i = 0; i < ntotalvars; ++i)
      nodecreated[i] = FALSE;

   /* create conflict graph */
   SCIP_CALL( SCIPdigraphCreate(&conshdlrdata->conflictgraph, cntsos) );

   /* set up hash map */
   SCIP_CALL( SCIPhashmapCreate(&conshdlrdata->varhash, SCIPblkmem(scip), cntsos) );

   /* for every SOS1 constraint */
   cntsos = 0;
   for (c = 0; c < nconss; ++c)
   {
      SCIP_CONSDATA* consdata;
      SCIP_VAR** vars;
      int nvars;

      assert( conss[c] != NULL );

      /* get constraint data field of the constraint */
      consdata = SCIPconsGetData(conss[c]);
      assert( consdata != NULL );

      /* get variables and number of variables of constraint */
      nvars = consdata->nvars;
      vars = consdata->vars;

      /* add edges to the conflict graph and create node data for each of its nodes */
      for (i = 0; i < nvars; ++i)
      {
         SCIP_VAR* var;

         var = vars[i];

         if ( SCIPvarGetStatus(var) != SCIP_VARSTATUS_FIXED )
         {
            int indi;

            indi = SCIPvarGetIndex(var);

            if ( ! nodecreated[indi] )
            {
               SCIP_NODEDATA* nodedata = NULL;

               /* insert node number to hash map */
               assert( ! SCIPhashmapExists(conshdlrdata->varhash, var) );
               SCIP_CALL( SCIPhashmapInsert(conshdlrdata->varhash, var, (void*) (size_t) cntsos) );/*lint !e571*/
               assert( cntsos == (int) (size_t) SCIPhashmapGetImage(conshdlrdata->varhash, var) );
               assert( SCIPhashmapExists(conshdlrdata->varhash, var) );

               /* create node data */
               SCIP_CALL( SCIPallocMemory(scip, &nodedata) );
               nodedata->var = var;
               nodedata->lbboundvar = NULL;
               nodedata->ubboundvar = NULL;
               nodedata->lbboundcoef = 0.0;
               nodedata->ubboundcoef = 0.0;
               nodedata->lbboundcomp = FALSE;
               nodedata->ubboundcomp = FALSE;

               /* set node data */
               SCIPdigraphSetNodeData(conshdlrdata->conflictgraph, (void*)nodedata, cntsos);

               /* mark node and var data of node as created and update SOS1 counter */
               nodecreated[indi] = TRUE;
               ++cntsos;
            }

            /* add edges to the conflict graph */
            for (j = i+1; j < nvars; ++j)
            {
               var = vars[j];

               if ( SCIPvarGetStatus(var) != SCIP_VARSTATUS_FIXED )
               {
                  int indj;

                  indj = SCIPvarGetIndex(var);

                  /* in case indi = indj the variable will be deleted in the presolving step */
                  if ( indi != indj )
                  {
                     /* arcs have to be added 'safe' */
                     SCIP_CALL( SCIPdigraphAddArcSafe(conshdlrdata->conflictgraph, nodeorig[indi], nodeorig[indj], NULL) );
                     SCIP_CALL( SCIPdigraphAddArcSafe(conshdlrdata->conflictgraph, nodeorig[indj], nodeorig[indi], NULL) );
                  }
               }
            }
         }
      }
   }

   /* set number of problem variables that are contained in at least one SOS1 constraint */
   conshdlrdata->nsos1vars = cntsos;

   /* free buffer arrays */
   SCIPfreeBufferArray(scip, &nodecreated);
   SCIPfreeBufferArray(scip, &nodeorig);

   /* sort successors in ascending order */
   for (j = 0; j < conshdlrdata->nsos1vars; ++j)
   {
      int nsucc;

      nsucc = SCIPdigraphGetNSuccessors(conshdlrdata->conflictgraph, j);
      SCIPsortInt(SCIPdigraphGetSuccessors(conshdlrdata->conflictgraph, j), nsucc);
   }

   return SCIP_OKAY;
}


/** free conflict graph, nodedata and hashmap */
static
SCIP_RETCODE freeConflictgraph(
   SCIP_CONSHDLRDATA*    conshdlrdata        /**< constraint handler data */
   )
{
   int j;

   /* for every SOS1 variable */
   for (j = 0; j < conshdlrdata->nsos1vars; ++j)
   {
      SCIP_NODEDATA* nodedata;

      /* get node data */
      nodedata = (SCIP_NODEDATA*)SCIPdigraphGetNodeData(conshdlrdata->conflictgraph, j);
      assert( nodedata != NULL );

      /* free node data */
      SCIPfreeMemory(scip, &nodedata);
      SCIPdigraphSetNodeData(conshdlrdata->conflictgraph, NULL, j);
   }

   /* free conflict graph and hash map */
   if ( conshdlrdata->conflictgraph != NULL )
   {
      assert( conshdlrdata->nsos1vars > 0 );
      assert( conshdlrdata->varhash != NULL );
      SCIPhashmapFree(&conshdlrdata->varhash);
      SCIPdigraphFree(&conshdlrdata->conflictgraph);
   }

   return SCIP_OKAY;
}


/* ---------------------------- constraint handler callback methods ----------------------*/

/** copy method for constraint handler plugins (called when SCIP copies plugins) */
static
SCIP_DECL_CONSHDLRCOPY(conshdlrCopySOS1)
{  /*lint --e{715}*/
   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );

   /* call inclusion method of constraint handler */
   SCIP_CALL( SCIPincludeConshdlrSOS1(scip) );

   *valid = TRUE;

   return SCIP_OKAY;
}


/** destructor of constraint handler to free constraint handler data (called when SCIP is exiting) */
static
SCIP_DECL_CONSFREE(consFreeSOS1)
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIPfreeMemory(scip, &conshdlrdata);

   return SCIP_OKAY;
}


/** solving process initialization method of constraint handler (called when branch and bound process is about to begin) */
static
SCIP_DECL_CONSINITSOL(consInitsolSOS1)
{  /*lint --e{715}*/
    SCIP_CONSHDLRDATA* conshdlrdata;

    assert( scip != NULL );
    assert( conshdlr != NULL );
    assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );

    conshdlrdata = SCIPconshdlrGetData(conshdlr);
    assert( conshdlrdata != NULL );

    conshdlrdata->nsos1vars = 0;
    conshdlrdata->varhash = NULL;

    if ( nconss > 0 )
    {
       /* initialize conflict graph and hashmap for SOS1 variables */
       SCIP_CALL( initConflictgraph(scip, conshdlrdata, conss, nconss) );

       /* add data to conflict graph nodes */
       SCIP_CALL( setNodeDataSOS1(scip, conshdlr, conshdlrdata, nconss, conshdlrdata->nsos1vars) );
    }
    return SCIP_OKAY;
}


/** solving process deinitialization method of constraint handler (called before branch and bound process data is freed) */
static
SCIP_DECL_CONSEXITSOL(consExitsolSOS1)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   int c;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );

   /* free graph for storing local conflicts */
   if ( conshdlrdata->localconflicts != NULL )
      SCIPdigraphFree(&conshdlrdata->localconflicts);
   assert( conshdlrdata->localconflicts == NULL );

   /* free tclique graph and tclique data */
   if( conshdlrdata->tcliquegraph != NULL )
   {
      assert( conshdlrdata->tcliquedata != NULL );
      SCIPfreeMemory(scip, &conshdlrdata->tcliquedata);
      tcliqueFree(&conshdlrdata->tcliquegraph);
   }
   assert(conshdlrdata->tcliquegraph == NULL);
   assert(conshdlrdata->tcliquedata == NULL);

   /* free conflict graph */
   if ( nconss > 0 && conshdlrdata->nsos1vars > 0 )
   {
      SCIP_CALL( freeConflictgraph(conshdlrdata) );
   }
   assert( conshdlrdata->conflictgraph == NULL );

   /* check each constraint */
   for (c = 0; c < nconss; ++c)
   {
      SCIP_CONSDATA* consdata;

      assert( conss != NULL );
      assert( conss[c] != NULL );
      consdata = SCIPconsGetData(conss[c]);
      assert( consdata != NULL );

      SCIPdebugMessage("Exiting SOS1 constraint <%s>.\n", SCIPconsGetName(conss[c]) );

      /* free rows */
      if ( consdata->rowub != NULL )
         SCIP_CALL( SCIPreleaseRow(scip, &consdata->rowub) );

      if ( consdata->rowlb != NULL )
         SCIP_CALL( SCIPreleaseRow(scip, &consdata->rowlb) );
   }

   return SCIP_OKAY;
}


/** frees specific constraint data */
static
SCIP_DECL_CONSDELETE(consDeleteSOS1)
{
   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( cons != NULL );
   assert( consdata != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );

   SCIPdebugMessage("Deleting SOS1 constraint <%s>.\n", SCIPconsGetName(cons) );

   /* drop events on transformed variables */
   if ( SCIPconsIsTransformed(cons) )
   {
      SCIP_CONSHDLRDATA* conshdlrdata;
      int j;

      /* get constraint handler data */
      conshdlrdata = SCIPconshdlrGetData(conshdlr);
      assert( conshdlrdata != NULL );
      assert( conshdlrdata->eventhdlr != NULL );

      for (j = 0; j < (*consdata)->nvars; ++j)
      {
         SCIP_CALL( SCIPdropVarEvent(scip, (*consdata)->vars[j], SCIP_EVENTTYPE_BOUNDCHANGED, conshdlrdata->eventhdlr,
               (SCIP_EVENTDATA*)*consdata, -1) );
      }
   }

   SCIPfreeBlockMemoryArray(scip, &(*consdata)->vars, (*consdata)->maxvars);
   if ( (*consdata)->weights != NULL )
   {
      SCIPfreeBlockMemoryArray(scip, &(*consdata)->weights, (*consdata)->maxvars);
   }

   /* free rows */
   if ( (*consdata)->rowub != NULL )
   {
      SCIP_CALL( SCIPreleaseRow(scip, &(*consdata)->rowub) );
   }
   if ( (*consdata)->rowlb != NULL )
   {
      SCIP_CALL( SCIPreleaseRow(scip, &(*consdata)->rowlb) );
   }
   assert( (*consdata)->rowub == NULL );
   assert( (*consdata)->rowlb == NULL );

   SCIPfreeBlockMemory(scip, consdata);

   return SCIP_OKAY;
}


/** transforms constraint data into data belonging to the transformed problem */
static
SCIP_DECL_CONSTRANS(consTransSOS1)
{
   SCIP_CONSDATA* consdata;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* sourcedata;
   char s[SCIP_MAXSTRLEN];
   int j;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );
   assert( sourcecons != NULL );
   assert( targetcons != NULL );

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );
   assert( conshdlrdata->eventhdlr != NULL );

   SCIPdebugMessage("Transforming SOS1 constraint: <%s>.\n", SCIPconsGetName(sourcecons) );

   /* get data of original constraint */
   sourcedata = SCIPconsGetData(sourcecons);
   assert( sourcedata != NULL );
   assert( sourcedata->nvars > 0 );
   assert( sourcedata->nvars <= sourcedata->maxvars );

   /* create constraint data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &consdata) );

   consdata->nvars = sourcedata->nvars;
   consdata->maxvars = sourcedata->nvars;
   consdata->rowub = NULL;
   consdata->rowlb = NULL;
   consdata->nfixednonzeros = 0;
   consdata->local = sourcedata->local;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &consdata->vars, consdata->nvars) );
   /* if weights were used */
   if ( sourcedata->weights != NULL )
   {
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &consdata->weights, sourcedata->weights, consdata->nvars) );
   }
   else
      consdata->weights = NULL;

   for (j = 0; j < sourcedata->nvars; ++j)
   {
      assert( sourcedata->vars[j] != 0 );
      SCIP_CALL( SCIPgetTransformedVar(scip, sourcedata->vars[j], &(consdata->vars[j])) );

      /* if variable is fixed to be nonzero */
      if ( SCIPisFeasPositive(scip, SCIPvarGetLbLocal(consdata->vars[j])) || SCIPisFeasNegative(scip, SCIPvarGetUbLocal(consdata->vars[j])) )
         ++(consdata->nfixednonzeros);
   }

   /* create transformed constraint with the same flags */
   (void) SCIPsnprintf(s, SCIP_MAXSTRLEN, "t_%s", SCIPconsGetName(sourcecons));
   SCIP_CALL( SCIPcreateCons(scip, targetcons, s, conshdlr, consdata,
         SCIPconsIsInitial(sourcecons), SCIPconsIsSeparated(sourcecons),
         SCIPconsIsEnforced(sourcecons), SCIPconsIsChecked(sourcecons),
         SCIPconsIsPropagated(sourcecons), SCIPconsIsLocal(sourcecons),
         SCIPconsIsModifiable(sourcecons), SCIPconsIsDynamic(sourcecons),
         SCIPconsIsRemovable(sourcecons), SCIPconsIsStickingAtNode(sourcecons)) );

   /* catch bound change events on variable */
   for (j = 0; j < consdata->nvars; ++j)
   {
      SCIP_CALL( SCIPcatchVarEvent(scip, consdata->vars[j], SCIP_EVENTTYPE_BOUNDCHANGED, conshdlrdata->eventhdlr,
            (SCIP_EVENTDATA*)consdata, NULL) );
   }

#ifdef SCIP_DEBUG
   if ( consdata->nfixednonzeros > 0 )
   {
      SCIPdebugMessage("constraint <%s> has %d variables fixed to be nonzero.\n", SCIPconsGetName(*targetcons),
         consdata->nfixednonzeros );
   }
#endif

   return SCIP_OKAY;
}


/** presolving method of constraint handler */
static
SCIP_DECL_CONSPRESOL(consPresolSOS1)
{  /*lint --e{715}*/
   int oldnfixedvars;
   int oldndelconss;
   int oldnupgdconss;
   int nremovedvars;
   SCIP_EVENTHDLR* eventhdlr;
   int c;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );
   assert( result != NULL );

   SCIPdebugMessage("Presolving SOS1 constraints.\n");

   *result = SCIP_DIDNOTRUN;
   oldnfixedvars = *nfixedvars;
   oldndelconss = *ndelconss;
   oldnupgdconss = *nupgdconss;
   nremovedvars = 0;

   /* only run if success if possible */
   if( nrounds == 0 || nnewfixedvars > 0 || nnewaggrvars > 0 )
   {
      /* get constraint handler data */
      assert( SCIPconshdlrGetData(conshdlr) != NULL );
      eventhdlr = SCIPconshdlrGetData(conshdlr)->eventhdlr;
      assert( eventhdlr != NULL );

      *result = SCIP_DIDNOTFIND;

      /* check each constraint */
      for (c = 0; c < nconss; ++c)
      {
         SCIP_CONSDATA* consdata;
         SCIP_CONS* cons;
         SCIP_Bool cutoff;
         SCIP_Bool success;

         assert( conss != NULL );
         assert( conss[c] != NULL );
         cons = conss[c];
         consdata = SCIPconsGetData(cons);

         assert( consdata != NULL );
         assert( consdata->nvars >= 0 );
         assert( consdata->nvars <= consdata->maxvars );
         assert( ! SCIPconsIsModifiable(cons) );

         /* perform one presolving round */
         SCIP_CALL( presolRoundSOS1(scip, cons, consdata, eventhdlr, &cutoff, &success, ndelconss, nupgdconss, nfixedvars, &nremovedvars) );

         if ( cutoff )
         {
            *result = SCIP_CUTOFF;
            return SCIP_OKAY;
         }

         if ( success )
            *result = SCIP_SUCCESS;
      }
   }
   (*nchgcoefs) += nremovedvars;

   SCIPdebugMessage("presolving fixed %d variables, removed %d variables, deleted %d constraints, and upgraded %d constraints.\n",
      *nfixedvars - oldnfixedvars, nremovedvars, *ndelconss - oldndelconss, *nupgdconss - oldnupgdconss);

   return SCIP_OKAY;
}


/** LP initialization method of constraint handler (called before the initial LP relaxation at a node is solved) */
static
SCIP_DECL_CONSINITLP(consInitlpSOS1)
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );

   /* checking for initial rows for SOS1 constraints */
   if( conshdlrdata->sepafromsos1 )
      SCIP_CALL( initsepaBoundInequalityFromSOS1Cons(scip, conshdlr, conshdlrdata, conss, nconss, NULL, FALSE, -1, NULL, NULL) );

   return SCIP_OKAY;
}


/** separation method of constraint handler for LP solutions */
static
SCIP_DECL_CONSSEPALP(consSepalpSOS1)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   int maxboundcuts;
   int ngen = 0;
   int depth;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( conss != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );
   assert( result != NULL );

   *result = SCIP_DIDNOTRUN;

   if ( nconss == 0 )
      return SCIP_OKAY;

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );

   /* check for boundcutsdepth < depth, maxboundcutsroot = 0 and maxboundcuts = 0 */
   depth = SCIPgetDepth(scip);
   if ( conshdlrdata->boundcutsdepth >= 0 && conshdlrdata->boundcutsdepth < depth )
      return SCIP_OKAY;

   /* only generate bound cuts if we are not close to terminating */
   if( SCIPisStopped(scip) )
      return SCIP_OKAY;

   /* determine maximal number of cuts*/
   if ( depth == 0 )
      maxboundcuts = conshdlrdata->maxboundcutsroot;
   else
      maxboundcuts = conshdlrdata->maxboundcuts;
   if ( maxboundcuts < 1 )
      return SCIP_OKAY;

   /* separate inequalities from SOS1 constraints */
   if( conshdlrdata->sepafromsos1 )
   {
      SCIP_CALL( initsepaBoundInequalityFromSOS1Cons(scip, conshdlr, conshdlrdata, conss, nconss, NULL, TRUE, maxboundcuts, &ngen, result) );
   }

   /* separate inequalities from the conflict graph */
   if( conshdlrdata->sepafromgraph )
   {
      SCIP_CALL( sepaBoundInequalitiesFromGraph(scip, conshdlr, conshdlrdata, NULL, maxboundcuts, &ngen, result) );
   }

   SCIPdebugMessage("Separated %d SOS1 constraints.\n", ngen);

   return SCIP_OKAY;
}


/** separation method of constraint handler for arbitrary primal solutions */
static
SCIP_DECL_CONSSEPASOL(consSepasolSOS1)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   int maxboundcuts;
   int ngen = 0;
   int depth;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( conss != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );
   assert( result != NULL );

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );

   *result = SCIP_DIDNOTRUN;

   if ( nconss == 0 )
      return SCIP_OKAY;

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );

   /* check for boundcutsdepth < depth, maxboundcutsroot = 0 and maxboundcuts = 0 */
   depth = SCIPgetDepth(scip);
   if ( conshdlrdata->boundcutsdepth >= 0 && conshdlrdata->boundcutsdepth < depth )
      return SCIP_OKAY;

   /* only generate bound cuts if we are not close to terminating */
   if( SCIPisStopped(scip) )
      return SCIP_OKAY;

   /* determine maximal number of cuts*/
   if ( depth == 0 )
      maxboundcuts = conshdlrdata->maxboundcutsroot;
   else
      maxboundcuts = conshdlrdata->maxboundcuts;
   if ( maxboundcuts < 1 )
      return SCIP_OKAY;

   /* separate inequalities from sos1 constraints */
   if( conshdlrdata->sepafromsos1 )
      SCIP_CALL( initsepaBoundInequalityFromSOS1Cons(scip, conshdlr, conshdlrdata, conss, nconss, sol, TRUE, maxboundcuts, &ngen, result) );

   /* separate inequalities from the conflict graph */
   if( conshdlrdata->sepafromgraph )
   {
      SCIP_CALL( sepaBoundInequalitiesFromGraph(scip, conshdlr, conshdlrdata, sol, maxboundcuts, &ngen, result) );
   }

   SCIPdebugMessage("Separated %d SOS1 constraints.\n", ngen);

   return SCIP_OKAY;
}


/** constraint enforcing method of constraint handler for LP solutions */
static
SCIP_DECL_CONSENFOLP(consEnfolpSOS1)
{  /*lint --e{715}*/
   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( conss != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );
   assert( result != NULL );

   SCIP_CALL( enforceSOS1(scip, conshdlr, nconss, conss, result) );

   return SCIP_OKAY;
}


/** constraint enforcing method of constraint handler for pseudo solutions */
static
SCIP_DECL_CONSENFOPS(consEnfopsSOS1)
{  /*lint --e{715}*/
   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( conss != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );
   assert( result != NULL );

   SCIP_CALL( enforceSOS1(scip, conshdlr, nconss, conss, result) );

   return SCIP_OKAY;
}


/** feasibility check method of constraint handler for integral solutions
 *
 *  We simply check whether at most one variable is nonzero in the given solution.
 */
static
SCIP_DECL_CONSCHECK(consCheckSOS1)
{  /*lint --e{715}*/
   int c;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( conss != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );
   assert( result != NULL );

   /* check each constraint */
   for (c = 0; c < nconss; ++c)
   {
      SCIP_CONSDATA* consdata;
      int j;
      int cnt;

      cnt = 0;
      assert( conss[c] != NULL );
      consdata = SCIPconsGetData(conss[c]);
      assert( consdata != NULL );
      SCIPdebugMessage("Checking SOS1 constraint <%s>.\n", SCIPconsGetName(conss[c]));

      /* check all variables */
      for (j = 0; j < consdata->nvars; ++j)
      {
         /* if variable is nonzero */
         if ( ! SCIPisFeasZero(scip, SCIPgetSolVal(scip, sol, consdata->vars[j])) )
         {
            ++cnt;

            /* if more than one variable is nonzero */
            if ( cnt > 1 )
            {
               SCIP_CALL( SCIPresetConsAge(scip, conss[c]) );
               *result = SCIP_INFEASIBLE;

               if ( printreason )
               {
                  int l;

                  SCIP_CALL( SCIPprintCons(scip, conss[c], NULL) );
                  SCIPinfoMessage(scip, NULL, ";\nviolation: ");

                  for (l = 0; l < consdata->nvars; ++l)
                  {
                     /* if variable is nonzero */
                     if ( ! SCIPisFeasZero(scip, SCIPgetSolVal(scip, sol, consdata->vars[l])) )
                     {
                        SCIPinfoMessage(scip, NULL, "<%s> = %.15g ",
                           SCIPvarGetName(consdata->vars[l]), SCIPgetSolVal(scip, sol, consdata->vars[l]));
                     }
                  }
                  SCIPinfoMessage(scip, NULL, "\n");
               }
               return SCIP_OKAY;
            }
         }
      }
   }
   *result = SCIP_FEASIBLE;

   return SCIP_OKAY;
}


/** domain propagation method of constraint handler */
static
SCIP_DECL_CONSPROP(consPropSOS1)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_DIGRAPH* conflictgraph;
   int ngen = 0;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( conss != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );
   assert( result != NULL );
   *result = SCIP_DIDNOTRUN;

   assert( SCIPisTransformed(scip) );

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );

   /* get conflict graph */
   conflictgraph = conshdlrdata->conflictgraph;

   /* if conflict graph propagation shall be used */
   if ( conshdlrdata->conflictprop && conflictgraph != NULL )
   {
      int nsos1vars;
      int j;

      /* get number of SOS1 variables */
      nsos1vars = conshdlrdata->nsos1vars;

      /* check each SOS1 variable */
      for (j = 0; j < nsos1vars; ++j)
      {
         SCIP_VAR* var;

         var = nodeGetVarSOS1(conflictgraph, j);
         SCIPdebugMessage("Propagating SOS1 variable <%s>.\n", SCIPvarGetName(var) );

         /* if zero is outside the domain of variable */
         if ( SCIPisFeasPositive(scip, SCIPvarGetLbLocal(var)) || SCIPisFeasNegative(scip, SCIPvarGetUbLocal(var)) )
         {
            SCIP_VAR* succvar;
            SCIP_Real lb;
            SCIP_Real ub;
            int* succ;
            int nsucc;
            int s;

            /* fix all neighbors in the conflict graph to zero */
            succ = SCIPdigraphGetSuccessors(conflictgraph, j);
            nsucc = SCIPdigraphGetNSuccessors(conflictgraph, j);
            for (s = 0; s < nsucc; ++s)
            {
               succvar = nodeGetVarSOS1(conflictgraph, succ[s]);
               lb = SCIPvarGetLbLocal(succvar);
               ub = SCIPvarGetUbLocal(succvar);

               if ( ! SCIPisFeasZero(scip, lb) || ! SCIPisFeasZero(scip, ub) )
               {
                  /* if variable cannot be nonzero */
                  if ( SCIPisFeasPositive(scip, lb) || SCIPisFeasNegative(scip, ub) )
                  {
                     *result = SCIP_CUTOFF;
                     return SCIP_OKAY;
                  }

                  /* directly fix variable if it is not multi-aggregated */
                  if ( SCIPvarGetStatus(succvar) != SCIP_VARSTATUS_MULTAGGR )
                  {
                     SCIP_Bool infeasible;
                     SCIP_Bool tightened;

                     SCIP_CALL( SCIPtightenVarLb(scip, succvar, 0.0, FALSE, &infeasible, &tightened) );
                     assert( ! infeasible );
                     if ( tightened )
                        ++ngen;

                     SCIP_CALL( SCIPtightenVarUb(scip, succvar, 0.0, FALSE, &infeasible, &tightened) );
                     assert( ! infeasible );
                     if ( tightened )
                        ++ngen;
                  }
               }
            }
         }
      }
   }

   /* if SOS1 constraint propagation shall be used */
   if ( conshdlrdata->sosconsprop || conflictgraph == NULL )
   {
      int c;

      /* check each constraint */
      for (c = 0; c < nconss; ++c)
      {
         SCIP_CONS* cons;
         SCIP_CONSDATA* consdata;
         SCIP_Bool cutoff;

         *result = SCIP_DIDNOTFIND;
         assert( conss[c] != NULL );
         cons = conss[c];
         consdata = SCIPconsGetData(cons);
         assert( consdata != NULL );
         SCIPdebugMessage("Propagating SOS1 constraint <%s>.\n", SCIPconsGetName(cons) );

         *result = SCIP_DIDNOTFIND;
         SCIP_CALL( propSOS1(scip, cons, consdata, &cutoff, &ngen) );
         if ( cutoff )
         {
            *result = SCIP_CUTOFF;
            return SCIP_OKAY;
         }
      }
   }

   SCIPdebugMessage("Propagated %d domains.\n", ngen);
   if ( ngen > 0 )
      *result = SCIP_REDUCEDDOM;

   return SCIP_OKAY;
}


/** propagation conflict resolving method of constraint handler
 *
 *  We check which bound changes were the reason for infeasibility. We
 *  use that @a inferinfo stores the index of the variable that has
 *  bounds that fix it to be nonzero (these bounds are the reason). */
static
SCIP_DECL_CONSRESPROP(consRespropSOS1)
{  /*lint --e{715}*/
   SCIP_CONSDATA* consdata;
   SCIP_VAR* var;

   assert( scip != NULL );
   assert( cons != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );
   assert( infervar != NULL );
   assert( bdchgidx != NULL );
   assert( result != NULL );

   *result = SCIP_DIDNOTFIND;
   SCIPdebugMessage("Propagation resolution method of SOS1 constraint <%s>.\n", SCIPconsGetName(cons));

   consdata = SCIPconsGetData(cons);
   assert( consdata != NULL );
   assert( 0 <= inferinfo && inferinfo < consdata->nvars );
   var = consdata->vars[inferinfo];
   assert( var != infervar );

   /* check if lower bound of var was the reason */
   if ( SCIPisFeasPositive(scip, SCIPvarGetLbAtIndex(var, bdchgidx, FALSE)) )
   {
      SCIP_CALL( SCIPaddConflictLb(scip, var, bdchgidx) );
      *result = SCIP_SUCCESS;
   }

   /* check if upper bound of var was the reason */
   if ( SCIPisFeasNegative(scip, SCIPvarGetUbAtIndex(var, bdchgidx, FALSE)) )
   {
      SCIP_CALL( SCIPaddConflictUb(scip, var, bdchgidx) );
      *result = SCIP_SUCCESS;
   }

   return SCIP_OKAY;
}


/** variable rounding lock method of constraint handler
 *
 *  Let lb and ub be the lower and upper bounds of a
 *  variable. Preprocessing usually makes sure that lb <= 0 <= ub.
 *
 *  - If lb < 0 then rounding down may violate the constraint.
 *  - If ub > 0 then rounding up may violated the constraint.
 *  - If lb > 0 or ub < 0 then the constraint is infeasible and we do
 *    not have to deal with it here.
 *  - If lb == 0 then rounding down does not violate the constraint.
 *  - If ub == 0 then rounding up does not violate the constraint.
 */
static
SCIP_DECL_CONSLOCK(consLockSOS1)
{
   SCIP_CONSDATA* consdata;
   SCIP_VAR** vars;
   int nvars;
   int j;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( cons != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );
   consdata = SCIPconsGetData(cons);
   assert( consdata != NULL );

   SCIPdebugMessage("Locking constraint <%s>.\n", SCIPconsGetName(cons));

   vars = consdata->vars;
   nvars = consdata->nvars;
   assert( vars != NULL );

   for (j = 0; j < nvars; ++j)
   {
      SCIP_VAR* var;
      var = vars[j];

      /* if lower bound is negative, rounding down may violate constraint */
      if ( SCIPisFeasNegative(scip, SCIPvarGetLbLocal(var)) )
      {
         SCIP_CALL( SCIPaddVarLocks(scip, var, nlockspos, nlocksneg) );
      }

      /* additionally: if upper bound is positive, rounding up may violate constraint */
      if ( SCIPisFeasPositive(scip, SCIPvarGetUbLocal(var)) )
      {
         SCIP_CALL( SCIPaddVarLocks(scip, var, nlocksneg, nlockspos) );
      }
   }

   return SCIP_OKAY;
}


/** constraint display method of constraint handler */
static
SCIP_DECL_CONSPRINT(consPrintSOS1)
{  /*lint --e{715}*/
   SCIP_CONSDATA* consdata;
   int j;

   assert( scip != NULL );
   assert( conshdlr != NULL );
   assert( cons != NULL );
   assert( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0 );

   consdata = SCIPconsGetData(cons);
   assert( consdata != NULL );

   for (j = 0; j < consdata->nvars; ++j)
   {
      if ( j > 0 )
         SCIPinfoMessage(scip, file, ", ");
      SCIP_CALL( SCIPwriteVarName(scip, file, consdata->vars[j], FALSE) );
      if ( consdata->weights == NULL )
         SCIPinfoMessage(scip, file, " (%d)", j+1);
      else
         SCIPinfoMessage(scip, file, " (%3.2f)", consdata->weights[j]);
   }

   return SCIP_OKAY;
}


/** constraint copying method of constraint handler */
static
SCIP_DECL_CONSCOPY(consCopySOS1)
{  /*lint --e{715}*/
   SCIP_CONSDATA* sourceconsdata;
   SCIP_VAR** sourcevars;
   SCIP_VAR** targetvars;
   SCIP_Real* sourceweights;
   SCIP_Real* targetweights;
   const char* consname;
   int nvars;
   int v;

   assert( scip != NULL );
   assert( sourcescip != NULL );
   assert( sourcecons != NULL );
   assert( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(sourcecons)), CONSHDLR_NAME) == 0 );

   *valid = TRUE;

   if ( name != NULL )
      consname = name;
   else
      consname = SCIPconsGetName(sourcecons);

   SCIPdebugMessage("Copying SOS1 constraint <%s> ...\n", consname);

   sourceconsdata = SCIPconsGetData(sourcecons);
   assert( sourceconsdata != NULL );

   /* get variables and weights of the source constraint */
   nvars = sourceconsdata->nvars;

   if ( nvars == 0 )
      return SCIP_OKAY;

   sourcevars = sourceconsdata->vars;
   assert( sourcevars != NULL );
   sourceweights = sourceconsdata->weights;
   assert( sourceweights != NULL );

   /* duplicate variable array */
   SCIP_CALL( SCIPallocBufferArray(sourcescip, &targetvars, nvars) );
   SCIP_CALL( SCIPduplicateBufferArray(sourcescip, &targetweights, sourceweights, nvars) );

   /* get copied variables in target SCIP */
   for( v = 0; v < nvars && *valid; ++v )
   {
      SCIP_CALL( SCIPgetVarCopy(sourcescip, scip, sourcevars[v], &(targetvars[v]), varmap, consmap, global, valid) );
   }

    /* only create the target constraint, if all variables could be copied */
   if( *valid )
   {
      SCIP_CALL( SCIPcreateConsSOS1(scip, cons, consname, nvars, targetvars, targetweights,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode) );
   }

   /* free buffer array */
   SCIPfreeBufferArray(sourcescip, &targetweights);
   SCIPfreeBufferArray(sourcescip, &targetvars);

   return SCIP_OKAY;
}


/** constraint parsing method of constraint handler */
static
SCIP_DECL_CONSPARSE(consParseSOS1)
{  /*lint --e{715}*/
   SCIP_VAR* var;
   SCIP_Real weight;
   const char* s;
   char* t;

   *success = TRUE;
   s = str;

   /* create empty SOS1 constraint */
   SCIP_CALL( SCIPcreateConsSOS1(scip, cons, name, 0, NULL, NULL, initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode) );

   /* loop through string */
   do
   {
      /* parse variable name */
      SCIP_CALL( SCIPparseVarName(scip, s, &var, &t) );
      s = t;

      /* skip until beginning of weight */
      while ( *s != '\0' && *s != '(' )
         ++s;

      if ( *s == '\0' )
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "Syntax error: expected weight at input: %s\n", s);
         *success = FALSE;
         return SCIP_OKAY;
      }
      /* skip '(' */
      ++s;

      /* find weight */
      weight = strtod(s, &t);
      if ( t == NULL )
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "Syntax error during parsing of the weight: %s\n", s);
         *success = FALSE;
         return SCIP_OKAY;
      }
      s = t;

      /* skip white space, ',', and ')' */
      while ( *s != '\0' && ( isspace((unsigned char)*s) ||  *s == ',' || *s == ')' ) )
         ++s;

      /* add variable */
      SCIP_CALL( SCIPaddVarSOS1(scip, *cons, var, weight) );
   }
   while ( *s != '\0' );

   return SCIP_OKAY;
}


/** constraint method of constraint handler which returns the variables (if possible) */
static
SCIP_DECL_CONSGETVARS(consGetVarsSOS1)
{  /*lint --e{715}*/
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   if( varssize < consdata->nvars )
      (*success) = FALSE;
   else
   {
      assert(vars != NULL);

      BMScopyMemoryArray(vars, consdata->vars, consdata->nvars);
      (*success) = TRUE;
   }

   return SCIP_OKAY;
}


/** constraint method of constraint handler which returns the number of variables (if possible) */
static
SCIP_DECL_CONSGETNVARS(consGetNVarsSOS1)
{  /*lint --e{715}*/
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   (*nvars) = consdata->nvars;
   (*success) = TRUE;

   return SCIP_OKAY;
}


/* ---------------- Callback methods of event handler ---------------- */

/* exec the event handler
 *
 * We update the number of variables fixed to be nonzero
 */
static
SCIP_DECL_EVENTEXEC(eventExecSOS1)
{
   SCIP_EVENTTYPE eventtype;
   SCIP_CONSDATA* consdata;
   SCIP_Real oldbound;
   SCIP_Real newbound;

   assert( eventhdlr != NULL );
   assert( eventdata != NULL );
   assert( strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0 );
   assert( event != NULL );

   consdata = (SCIP_CONSDATA*)eventdata;
   assert( consdata != NULL );
   assert( 0 <= consdata->nfixednonzeros && consdata->nfixednonzeros <= consdata->nvars );

   oldbound = SCIPeventGetOldbound(event);
   newbound = SCIPeventGetNewbound(event);

   eventtype = SCIPeventGetType(event);
   switch ( eventtype )
   {
   case SCIP_EVENTTYPE_LBTIGHTENED:
      /* if variable is now fixed to be nonzero */
      if ( ! SCIPisFeasPositive(scip, oldbound) && SCIPisFeasPositive(scip, newbound) )
         ++(consdata->nfixednonzeros);
      break;
   case SCIP_EVENTTYPE_UBTIGHTENED:
      /* if variable is now fixed to be nonzero */
      if ( ! SCIPisFeasNegative(scip, oldbound) && SCIPisFeasNegative(scip, newbound) )
         ++(consdata->nfixednonzeros);
      break;
   case SCIP_EVENTTYPE_LBRELAXED:
      /* if variable is not fixed to be nonzero anymore */
      if ( SCIPisFeasPositive(scip, oldbound) && ! SCIPisFeasPositive(scip, newbound) )
         --(consdata->nfixednonzeros);
      break;
   case SCIP_EVENTTYPE_UBRELAXED:
      /* if variable is not fixed to be nonzero anymore */
      if ( SCIPisFeasNegative(scip, oldbound) && ! SCIPisFeasNegative(scip, newbound) )
         --(consdata->nfixednonzeros);
      break;
   default:
      SCIPerrorMessage("invalid event type.\n");
      return SCIP_INVALIDDATA;
   }
   assert( 0 <= consdata->nfixednonzeros && consdata->nfixednonzeros <= consdata->nvars );

   SCIPdebugMessage("changed bound of variable <%s> from %f to %f (nfixednonzeros: %d).\n", SCIPvarGetName(SCIPeventGetVar(event)),
                    oldbound, newbound, consdata->nfixednonzeros);

   return SCIP_OKAY;
}


/* ---------------- Constraint specific interface methods ---------------- */

/** creates the handler for SOS1 constraints and includes it in SCIP */
SCIP_RETCODE SCIPincludeConshdlrSOS1(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSHDLR* conshdlr;

   /* create constraint handler data */
   SCIP_CALL( SCIPallocMemory(scip, &conshdlrdata) );
   conshdlrdata->branchsos = TRUE;
   conshdlrdata->eventhdlr = NULL;
   conshdlrdata->conflictgraph = NULL;
   conshdlrdata->localconflicts = NULL;
   conshdlrdata->isconflocal = FALSE;
   conshdlrdata->nboundcuts = 0;
   conshdlrdata->tcliquegraph = NULL;
   conshdlrdata->tcliquedata = NULL;

   /* create event handler for bound change events */
   SCIP_CALL( SCIPincludeEventhdlrBasic(scip, &conshdlrdata->eventhdlr, EVENTHDLR_NAME, EVENTHDLR_DESC, eventExecSOS1, NULL) );
   if ( conshdlrdata->eventhdlr == NULL )
   {
      SCIPerrorMessage("event handler for SOS1 constraints not found.\n");
      return SCIP_PLUGINNOTFOUND;
   }

   /* include constraint handler */
   SCIP_CALL( SCIPincludeConshdlrBasic(scip, &conshdlr, CONSHDLR_NAME, CONSHDLR_DESC,
         CONSHDLR_ENFOPRIORITY, CONSHDLR_CHECKPRIORITY, CONSHDLR_EAGERFREQ, CONSHDLR_NEEDSCONS,
         consEnfolpSOS1, consEnfopsSOS1, consCheckSOS1, consLockSOS1, conshdlrdata) );
   assert(conshdlr != NULL);

   /* set non-fundamental callbacks via specific setter functions */
   SCIP_CALL( SCIPsetConshdlrCopy(scip, conshdlr, conshdlrCopySOS1, consCopySOS1) );
   SCIP_CALL( SCIPsetConshdlrDelete(scip, conshdlr, consDeleteSOS1) );
   SCIP_CALL( SCIPsetConshdlrExitsol(scip, conshdlr, consExitsolSOS1) );
   SCIP_CALL( SCIPsetConshdlrInitsol(scip, conshdlr, consInitsolSOS1) );
   SCIP_CALL( SCIPsetConshdlrFree(scip, conshdlr, consFreeSOS1) );
   SCIP_CALL( SCIPsetConshdlrGetVars(scip, conshdlr, consGetVarsSOS1) );
   SCIP_CALL( SCIPsetConshdlrGetNVars(scip, conshdlr, consGetNVarsSOS1) );
   SCIP_CALL( SCIPsetConshdlrInitlp(scip, conshdlr, consInitlpSOS1) );
   SCIP_CALL( SCIPsetConshdlrParse(scip, conshdlr, consParseSOS1) );
   SCIP_CALL( SCIPsetConshdlrPresol(scip, conshdlr, consPresolSOS1, CONSHDLR_MAXPREROUNDS, CONSHDLR_DELAYPRESOL) );
   SCIP_CALL( SCIPsetConshdlrPrint(scip, conshdlr, consPrintSOS1) );
   SCIP_CALL( SCIPsetConshdlrProp(scip, conshdlr, consPropSOS1, CONSHDLR_PROPFREQ, CONSHDLR_DELAYPROP, CONSHDLR_PROP_TIMING) );
   SCIP_CALL( SCIPsetConshdlrResprop(scip, conshdlr, consRespropSOS1) );
   SCIP_CALL( SCIPsetConshdlrSepa(scip, conshdlr, consSepalpSOS1, consSepasolSOS1, CONSHDLR_SEPAFREQ, CONSHDLR_SEPAPRIORITY, CONSHDLR_DELAYSEPA) );
   SCIP_CALL( SCIPsetConshdlrTrans(scip, conshdlr, consTransSOS1) );

   /* add SOS1 constraint handler parameters */

   /* propagation parameters */
   SCIP_CALL( SCIPaddBoolParam(scip, "constraints/"CONSHDLR_NAME"/conflictprop",
         "whether to use conflict graph propagation",
         &conshdlrdata->conflictprop, TRUE, DEFAULT_CONFLICTPROP, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "constraints/"CONSHDLR_NAME"/sosconsprop",
         "whether to use SOS1 constraint propagation",
         &conshdlrdata->sosconsprop, TRUE, DEFAULT_SOSCONSPROP, NULL, NULL) );

   /* branching parameters */
   SCIP_CALL( SCIPaddBoolParam(scip, "constraints/"CONSHDLR_NAME"/branchsos",
         "Use SOS1 branching in enforcing (otherwise leave decision to branching rules)?",
         &conshdlrdata->branchsos, FALSE, TRUE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "constraints/"CONSHDLR_NAME"/branchnonzeros",
         "Branch on SOS constraint with most number of nonzeros?",
         &conshdlrdata->branchnonzeros, FALSE, FALSE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "constraints/"CONSHDLR_NAME"/branchweight",
         "Branch on SOS cons. with highest nonzero-variable weight for branching (needs branchnonzeros = false)?",
         &conshdlrdata->branchweight, FALSE, FALSE, NULL, NULL) );

   /* separation parameters */
   SCIP_CALL( SCIPaddBoolParam(scip, "constraints/"CONSHDLR_NAME"/sepafromsos1",
         "if TRUE separate bound inequalities from initial SOS1 constraints",
         &conshdlrdata->sepafromsos1, TRUE, DEFAULT_SEPAFROMSOS1, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "constraints/"CONSHDLR_NAME"/sepafromgraph",
         "if TRUE separate bound inequalities from the conflict graph",
         &conshdlrdata->sepafromgraph, TRUE, DEFAULT_SEPAFROMGRAPH, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "constraints/"CONSHDLR_NAME"/boundcutsdepth",
         "node depth of separating bound cuts (-1: no limit)",
         &conshdlrdata->boundcutsdepth, TRUE, DEFAULT_BOUNDCUTSDEPTH, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "constraints/"CONSHDLR_NAME"/maxboundcuts",
         "maximal number of bound cuts separated per branching node",
         &conshdlrdata->maxboundcuts, TRUE, DEFAULT_MAXBOUNDCUTS, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "constraints/"CONSHDLR_NAME"/maxboundcutsroot",
         "maximal number of bound cuts separated per iteration in the root node",
         &conshdlrdata->maxboundcutsroot, TRUE, DEFAULT_MAXBOUNDCUTSROOT, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "constraints/"CONSHDLR_NAME"/strthenboundcuts",
         "if TRUE then bound cuts are strengthened in case bound variables are available",
         &conshdlrdata->strthenboundcuts, TRUE, DEFAULT_STRTHENBOUNDCUTS, NULL, NULL) );

   return SCIP_OKAY;
}


/** creates and captures a SOS1 constraint
 *
 *  We set the constraint to not be modifable. If the weights are non
 *  NULL, the variables are ordered according to these weights (in
 *  ascending order).
 *
 *  @note the constraint gets captured, hence at one point you have to release it using the method SCIPreleaseCons()
 */
SCIP_RETCODE SCIPcreateConsSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   int                   nvars,              /**< number of variables in the constraint */
   SCIP_VAR**            vars,               /**< array with variables of constraint entries */
   SCIP_Real*            weights,            /**< weights determining the variable order, or NULL if natural order should be used */
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
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSDATA* consdata;
   SCIP_Bool modifiable;
   SCIP_Bool transformed;
   int v;

   modifiable = FALSE;

   /* find the SOS1 constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   if ( conshdlr == NULL )
   {
      SCIPerrorMessage("<%s> constraint handler not found\n", CONSHDLR_NAME);
      return SCIP_PLUGINNOTFOUND;
   }

   /* are we in the transformed problem? */
   transformed = SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED;

   /* create constraint data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &consdata) );
   consdata->vars = NULL;
   consdata->nvars = nvars;
   consdata->maxvars = nvars;
   consdata->rowub = NULL;
   consdata->rowlb = NULL;
   consdata->nfixednonzeros = transformed ? 0 : -1;
   consdata->weights = NULL;
   consdata->local = local;

   if ( nvars > 0 )
   {
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &consdata->vars, vars, nvars) );

      /* check weights */
      if ( weights != NULL )
      {
         /* store weights */
         SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &consdata->weights, weights, nvars) );

         /* sort variables - ascending order */
         SCIPsortRealPtr(consdata->weights, (void**)consdata->vars, nvars);
      }
   }
   else
   {
      assert( weights == NULL );
   }

   /* branching on multiaggregated variables does not seem to work well, so avoid it */
   for (v = 0; v < nvars; ++v)
      SCIP_CALL( SCIPmarkDoNotMultaggrVar(scip, consdata->vars[v]) );

   /* create constraint */
   SCIP_CALL( SCIPcreateCons(scip, cons, name, conshdlr, consdata, initial, separate, enforce, check, propagate,
         local, modifiable, dynamic, removable, stickingatnode) );
   assert(transformed == SCIPconsIsTransformed(*cons));

   /* replace original variables by transformed variables in transformed constraint, add locks, and catch events */
   for( v = nvars - 1; v >= 0; --v )
   {
      /* always use transformed variables in transformed constraints */
      if ( transformed )
      {
         SCIP_CALL( SCIPgetTransformedVar(scip, consdata->vars[v], &(consdata->vars[v])) );
      }
      assert( consdata->vars[v] != NULL );
      assert( transformed == SCIPvarIsTransformed(consdata->vars[v]) );

      /* handle the new variable */
      SCIP_CALL( handleNewVariableSOS1(scip, *cons, consdata, consdata->vars[v], transformed) );
   }

   return SCIP_OKAY;
}


/** creates and captures a SOS1 constraint with all constraint flags set to their default values.
 *
 *  @warning Do NOT set the constraint to be modifiable manually, because this might lead
 *  to wrong results as the variable array will not be resorted
 *
 *  @note the constraint gets captured, hence at one point you have to release it using the method SCIPreleaseCons()
 */
SCIP_RETCODE SCIPcreateConsBasicSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   int                   nvars,              /**< number of variables in the constraint */
   SCIP_VAR**            vars,               /**< array with variables of constraint entries */
   SCIP_Real*            weights             /**< weights determining the variable order, or NULL if natural order should be used */
   )
{
   SCIP_CALL( SCIPcreateConsSOS1( scip, cons, name, nvars, vars, weights, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE) );

   return SCIP_OKAY;
}


/** adds variable to SOS1 constraint, the position is determined by the given weight */
SCIP_RETCODE SCIPaddVarSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_VAR*             var,                /**< variable to add to the constraint */
   SCIP_Real             weight              /**< weight determining position of variable */
   )
{
   assert( scip != NULL );
   assert( var != NULL );
   assert( cons != NULL );

   SCIPdebugMessage("adding variable <%s> to constraint <%s> with weight %g\n", SCIPvarGetName(var), SCIPconsGetName(cons), weight);

   if ( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not an SOS1 constraint.\n");
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( addVarSOS1(scip, cons, var, weight) );

   return SCIP_OKAY;
}


/** appends variable to SOS1 constraint */
SCIP_RETCODE SCIPappendVarSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_VAR*             var                 /**< variable to add to the constraint */
   )
{
   assert( scip != NULL );
   assert( var != NULL );
   assert( cons != NULL );

   SCIPdebugMessage("appending variable <%s> to constraint <%s>\n", SCIPvarGetName(var), SCIPconsGetName(cons));

   if ( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not an SOS1 constraint.\n");
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( appendVarSOS1(scip, cons, var) );

   return SCIP_OKAY;
}


/** gets number of variables in SOS1 constraint */
int SCIPgetNVarsSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   )
{
   SCIP_CONSDATA* consdata;

   assert( scip != NULL );
   assert( cons != NULL );

   if ( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not an SOS1 constraint.\n");
      SCIPABORT();
      return -1;  /*lint !e527*/
   }

   consdata = SCIPconsGetData(cons);
   assert( consdata != NULL );

   return consdata->nvars;
}


/** gets array of variables in SOS1 constraint */
SCIP_VAR** SCIPgetVarsSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   )
{
   SCIP_CONSDATA* consdata;

   assert( scip != NULL );
   assert( cons != NULL );

   if ( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not an SOS1 constraint.\n");
      SCIPABORT();
      return NULL;  /*lint !e527*/
   }

   consdata = SCIPconsGetData(cons);
   assert( consdata != NULL );

   return consdata->vars;
}


/** gets array of weights in SOS1 constraint (or NULL if not existent) */
SCIP_Real* SCIPgetWeightsSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   )
{
   SCIP_CONSDATA* consdata;

   assert( scip != NULL );
   assert( cons != NULL );

   if ( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not an SOS1 constraint.\n");
      SCIPABORT();
      return NULL;  /*lint !e527*/
   }

   consdata = SCIPconsGetData(cons);
   assert( consdata != NULL );

   return consdata->weights;
}


/** gets conflict graph of SOS1 constraints (or NULL if not existent)
 *
 *  Note: The conflict graph is globally valid; local changes are not taken into account.
 */
SCIP_DIGRAPH* SCIPgetConflictgraphSOS1(
   SCIP_CONSHDLR*        conshdlr            /**< SOS1 constraint handler */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert( conshdlr != NULL );

   if ( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("not an SOS1 constraint handler.\n");
      SCIPABORT();
   }
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );

   return conshdlrdata->conflictgraph;
}


/** gets number of problem variables that are involved in at least one SOS1 constraint */
int SCIPgetNSOS1Vars(
   SCIP_CONSHDLR*        conshdlr            /**< SOS1 constraint handler */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert( conshdlr != NULL );

   if ( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("not an SOS1 constraint handler.\n");
      SCIPABORT();
   }
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );

   return conshdlrdata->nsos1vars;
}


/** returns whether variable is involved in an SOS1 constraint */
SCIP_Bool varIsSOS1(
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_VAR*             var                 /**< variable */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert( var != NULL );
   assert( conshdlr != NULL );

   if ( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("not an SOS1 constraint handler.\n");
      SCIPABORT();
   }
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );

   if ( conshdlrdata->varhash == NULL || ! SCIPhashmapExists(conshdlrdata->varhash, var) )
      return FALSE;

   return TRUE;
}


/** returns SOS1 index of variable or -1 if variable is not involved in an SOS1 constraint */
int varGetNodeSOS1(
   SCIP_CONSHDLR*        conshdlr,            /**< SOS1 constraint handler */
   SCIP_VAR*             var                  /**< variable */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert( conshdlr != NULL );
   assert( var != NULL );

   if ( strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("not an SOS1 constraint handler.\n");
      SCIPABORT();
   }
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert( conshdlrdata != NULL );

   if ( ! SCIPhashmapExists(conshdlrdata->varhash, var) )
      return -1;

   return (int) (size_t) SCIPhashmapGetImage(conshdlrdata->varhash, var);
}


/** returns variable that belongs to a given node from the conflictgraph */
SCIP_VAR* nodeGetVarSOS1(
   SCIP_DIGRAPH*         conflictgraph,      /**< conflict graph */
   int                   node                /**< node from the conflict graph */
   )
{
   SCIP_NODEDATA* nodedata;

   assert( conflictgraph != NULL );
   assert( node >= 0 && node < SCIPdigraphGetNNodes(conflictgraph) );

   /* get node data */
   nodedata = (SCIP_NODEDATA*)SCIPdigraphGetNodeData(conflictgraph, node);

   if ( nodedata == NULL )
   {
      SCIPerrorMessage("variable is not assigned to an index.\n");
      SCIPABORT();
   }

   return nodedata->var;
}


/** gets (variable) lower bound value of current LP relaxation solution for a given node from the conflict graph */
SCIP_Real SCIPnodeGetSolvalVarboundLbSOS1(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_DIGRAPH*         conflictgraph,      /**< conflict graph */
   SCIP_SOL*         	 sol,                /**< primal solution, or NULL for current LP/pseudo solution */
   int                   node                /**< node of the conflict graph */
   )
{
   SCIP_NODEDATA* nodedata;

   assert( scip != NULL );
   assert( conflictgraph != NULL );
   assert( node >= 0 && node < SCIPdigraphGetNNodes(conflictgraph) );

   /* get node data */
   nodedata = (SCIP_NODEDATA*)SCIPdigraphGetNodeData(conflictgraph, node);
   assert( nodedata != NULL );

   /* if variable is not involved in a variable upper bound constraint */
   if ( nodedata->lbboundvar == NULL || ! nodedata->lbboundcomp )
      return SCIPvarGetLbLocal(nodedata->var);

   return nodedata->lbboundcoef * SCIPgetSolVal(scip, sol, nodedata->lbboundvar);
}


/** gets (variable) upper bound value of current LP relaxation solution for a given node from the conflict graph */
SCIP_Real SCIPnodeGetSolvalVarboundUbSOS1(
   SCIP*                 scip,               /**< SCIP pointer */
   SCIP_DIGRAPH*         conflictgraph,      /**< conflict graph */
   SCIP_SOL*         	 sol,                /**< primal solution, or NULL for current LP/pseudo solution */
   int                   node                /**< node of the conflict graph */
   )
{
   SCIP_NODEDATA* nodedata;

   assert( scip != NULL );
   assert( conflictgraph != NULL );
   assert( node >= 0 && node < SCIPdigraphGetNNodes(conflictgraph) );

   /* get node data */
   nodedata = (SCIP_NODEDATA*)SCIPdigraphGetNodeData(conflictgraph, node);
   assert( nodedata != NULL );

   /* if variable is not involved in a variable upper bound constraint */
   if ( nodedata->ubboundvar == NULL || ! nodedata->ubboundcomp )
      return SCIPvarGetUbLocal(nodedata->var);

   return nodedata->ubboundcoef * SCIPgetSolVal(scip, sol, nodedata->ubboundvar);
}

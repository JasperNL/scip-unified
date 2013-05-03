/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2013 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   scip.h
 * @ingroup PUBLICMETHODS
 * @brief  SCIP callable library
 * @author Tobias Achterberg
 * @author Timo Berthold
 * @author Thorsten Koch
 * @author Alexander Martin
 * @author Marc Pfetsch
 * @author Kati Wolter
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_SCIP_H__
#define __SCIP_SCIP_H__


#include <stdio.h>

#include "scip/def.h"
#include "blockmemshell/memory.h"
#include "scip/type_retcode.h"
#include "scip/type_result.h"
#include "scip/type_clock.h"
#include "scip/type_misc.h"
#include "scip/type_timing.h"
#include "scip/type_paramset.h"
#include "scip/type_event.h"
#include "scip/type_lp.h"
#include "scip/type_nlp.h"
#include "scip/type_var.h"
#include "scip/type_prob.h"
#include "scip/type_tree.h"
#include "scip/type_scip.h"

#include "scip/type_branch.h"
#include "scip/type_conflict.h"
#include "scip/type_cons.h"
#include "scip/type_dialog.h"
#include "scip/type_disp.h"
#include "scip/type_heur.h"
#include "scip/type_history.h"
#include "scip/type_nodesel.h"
#include "scip/type_presol.h"
#include "scip/type_pricer.h"
#include "scip/type_reader.h"
#include "scip/type_relax.h"
#include "scip/type_sepa.h"
#include "scip/type_prop.h"
#include "nlpi/type_nlpi.h"

/* include public interfaces, s.t. the user only needs to include scip.h */
#include "scip/pub_branch.h"
#include "scip/pub_conflict.h"
#include "scip/pub_cons.h"
#include "scip/pub_cutpool.h"
#include "scip/pub_dialog.h"
#include "scip/pub_disp.h"
#include "scip/pub_event.h"
#include "scip/pub_fileio.h"
#include "scip/pub_heur.h"
#include "scip/pub_history.h"
#include "scip/pub_implics.h"
#include "scip/pub_lp.h"
#include "scip/pub_nlp.h"
#include "scip/pub_message.h"
#include "scip/pub_misc.h"
#include "scip/pub_nodesel.h"
#include "scip/pub_paramset.h"
#include "scip/pub_presol.h"
#include "scip/pub_pricer.h"
#include "scip/pub_reader.h"
#include "scip/pub_relax.h"
#include "scip/pub_sepa.h"
#include "scip/pub_prop.h"
#include "scip/pub_sol.h"
#include "scip/pub_tree.h"
#include "scip/pub_var.h"
#include "scip/lpi.h"
#include "nlpi/pub_expr.h"


/* In debug mode, we include the SCIP's structure in scip.c, such that no one can access
 * this structure except the interface methods in scip.c.
 * In optimized mode, the structure is included in scip.h, because some of the methods
 * are implemented as defines for performance reasons (e.g. the numerical comparisons).
 * Additionally, the internal "set.h" is included, such that the defines in set.h are
 * available in optimized mode.
 */
#ifdef NDEBUG
#include "scip/struct_scip.h"
#include "scip/set.h"
#include "scip/tree.h"
#include "scip/misc.h"
#include "scip/var.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * miscellaneous methods
 */

/**@name Miscellaneous Methods */
/**@{ */

/** returns complete SCIP version number in the format "major . minor tech"
 *
 *  @return complete SCIP version
 */
EXTERN
SCIP_Real SCIPversion(
   void
   );

/** returns SCIP major version
 *
 *  @return major SCIP version
 */
EXTERN
int SCIPmajorVersion(
   void
   );

/** returns SCIP minor version
 *
 *  @return minor SCIP version
 */
EXTERN
int SCIPminorVersion(
   void
   );

/** returns SCIP technical version
 *
 *  @return technical SCIP version
 */
EXTERN
int SCIPtechVersion(
   void
   );

/** returns SCIP sub version number
 *
 *  @return subversion SCIP version
 */
EXTERN
int SCIPsubversion(
   void
   );

/** prints a version information line to a file stream via the message handler system
 *
 *  @note If the message handler is set to a NULL pointer nothing will be printed
 */
EXTERN
void SCIPprintVersion(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/** prints error message for the given SCIP_RETCODE via the error prints method */
EXTERN
void SCIPprintError(
   SCIP_RETCODE          retcode             /**< SCIP return code causing the error */
   );

/** update statistical information when a new solution was found */
EXTERN
void SCIPstoreSolutionGap(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@} */




/*
 * general SCIP methods
 */

/**@name General SCIP Methods */
/**@{ */

/** creates and initializes SCIP data structures
 *
 *  @note The SCIP default message handler is installed. Use the method SCIPsetMessagehdlr() to install your own
 *        message handler or SCIPsetMessagehdlrLogfile() and SCIPsetMessagehdlrQuiet() to write into a log
 *        file and turn off/on the display output, respectively.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @post After calling this method @p scip reached the solving stage \ref SCIP_STAGE_INIT
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcreate(
   SCIP**                scip                /**< pointer to SCIP data structure */
   );

/** frees SCIP data structures
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_FREE
 *
 *  @post After calling this method \SCIP reached the solving stage \ref SCIP_STAGE_FREE
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPfree(
   SCIP**                scip                /**< pointer to SCIP data structure */
   );

/** returns current stage of SCIP
 *
 *  @return the current SCIP stage
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_STAGE SCIPgetStage(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** outputs SCIP stage and solution status if applicable via the message handler
 *
 *  @note If the message handler is set to a NULL pointer nothing will be printed
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPprintStage(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/** gets solution status
 *
 *  @return SCIP solution status
 *
 *  See \ref SCIP_Status "SCIP_STATUS" for a complete list of all possible solving status.
 */
EXTERN
SCIP_STATUS SCIPgetStatus(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** outputs solution status
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  See \ref SCIP_Status "SCIP_STATUS" for a complete list of all possible solving status.
 */
EXTERN
SCIP_RETCODE SCIPprintStatus(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/** returns whether the current stage belongs to the transformed problem space
 *
 *  @return Returns TRUE if the \SCIP instance is transformed, otherwise FALSE
 */
EXTERN
SCIP_Bool SCIPisTransformed(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether the solution process should be probably correct
 *
 *  @note This feature is not supported yet!
 *
 *  @return Returns TRUE if \SCIP is exact solving mode, otherwise FALSE
 */
EXTERN
SCIP_Bool SCIPisExactSolve(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether the presolving process would be finished given no more presolving reductions are found in this
 *  presolving round
 *
 *  Checks whether the number of presolving rounds is not exceeded and the presolving reductions found in the current
 *  presolving round suffice to trigger another presolving round.
 *
 *  @note if subsequent presolvers find more reductions, presolving might continue even if the method returns FALSE
 *  @note does not check whether infeasibility or unboundedness was already detected in presolving (which would result
 *        in presolving being stopped although the method returns TRUE)
 *
 *  @return Returns TRUE if presolving is finished if no further reductions are detected
 */
EXTERN
SCIP_Bool SCIPisPresolveFinished(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether the user pressed CTRL-C to interrupt the solving process
 *
 *  @return Returns TRUE if Ctrl-C was pressed, otherwise FALSE.
 */
EXTERN
SCIP_Bool SCIPpressedCtrlC(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether the solving process should be / was stopped before proving optimality;
 *  if the solving process should be / was stopped, the status returned by SCIPgetStatus() yields
 *  the reason for the premature abort
 *
 *  @return Returns TRUE if solving process is stopped/interrupted, otherwise FALSE.
 */
EXTERN
SCIP_Bool SCIPisStopped(
   SCIP*                 scip                /**< SCIP data structure */
   );


/**@} */



/*
 * message output methods
 */

/**@name Message Output Methods */
/**@{ */

/** installs the given message handler, such that all messages are passed to this handler. A messages handler can be
 *  created via SCIPmessagehdlrCreate().
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note The currently installed messages handler gets freed if this SCIP instance is its last user (w.r.t. capture/release).
 */
EXTERN
SCIP_RETCODE SCIPsetMessagehdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_MESSAGEHDLR*     messagehdlr         /**< message handler to install, or NULL to suppress all output */
   );

/** returns the currently installed message handler
 *
 *  @return the currently installed message handler, or NULL if messages are currently suppressed
 */
EXTERN
SCIP_MESSAGEHDLR* SCIPgetMessagehdlr(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the log file name for the currently installed message handler */
EXTERN
void SCIPsetMessagehdlrLogfile(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename            /**< name of log file, or NULL (no log) */
   );

/** sets the currently installed message handler to be quiet (or not) */
EXTERN
void SCIPsetMessagehdlrQuiet(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             quiet               /**< should screen messages be suppressed? */
   );

/** prints a warning message via the message handler */
EXTERN
void SCIPwarningMessage(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           formatstr,          /**< format string like in printf() function */
   ...                                       /**< format arguments line in printf() function */
   );

/** prints a dialog message that requests user interaction or is a direct response to a user interactive command */
EXTERN
void SCIPdialogMessage(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< file stream to print into, or NULL for stdout */
   const char*           formatstr,          /**< format string like in printf() function */
   ...                                       /**< format arguments line in printf() function */
   );

/** prints a message */
EXTERN
void SCIPinfoMessage(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< file stream to print into, or NULL for stdout */
   const char*           formatstr,          /**< format string like in printf() function */
   ...                                       /**< format arguments line in printf() function */
   );

/** prints a message depending on the verbosity level */
EXTERN
void SCIPverbMessage(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VERBLEVEL        msgverblevel,       /**< verbosity level of this message */
   FILE*                 file,               /**< file stream to print into, or NULL for stdout */
   const char*           formatstr,          /**< format string like in printf() function */
   ...                                       /**< format arguments line in printf() function */
   );

/** returns the current message verbosity level
 *
 *  @return message verbosity level of SCIP
 *
 *  @see \ref SCIP_VerbLevel "SCIP_VERBLEVEL" for a list of all verbosity levels
 */
EXTERN
SCIP_VERBLEVEL SCIPgetVerbLevel(
   SCIP*                 scip                /**< SCIP data structure */
   );


/**@} */




/*
 * SCIP copy methods
 */

/**@name Copy Methods */
/**@{ */

/** copies plugins from sourcescip to targetscip; in case that a constraint handler which does not need constraints
 *  cannot be copied, valid will return FALSE. All plugins can declare that, if their copy process failed, the
 *  copied SCIP instance might not represent the same problem semantics as the original.
 *  Note that in this case dual reductions might be invalid.
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *        Also, 'passmessagehdlr' should be set to FALSE.
 *  @note Do not change the source SCIP environment during the copying process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_FREE
 *
 *  @post After calling this method targetscip reaches one of the following stages depending on if and when the solution
 *        process was interrupted:
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note sourcescip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcopyPlugins(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_Bool             copyreaders,        /**< should the file readers be copied */
   SCIP_Bool             copypricers,        /**< should the variable pricers be copied */
   SCIP_Bool             copyconshdlrs,      /**< should the constraint handlers be copied */
   SCIP_Bool             copyconflicthdlrs,  /**< should the conflict handlers be copied */
   SCIP_Bool             copypresolvers,     /**< should the presolvers be copied */
   SCIP_Bool             copyrelaxators,     /**< should the relaxation handler be copied */
   SCIP_Bool             copyseparators,     /**< should the separators be copied */
   SCIP_Bool             copypropagators,    /**< should the propagators be copied */
   SCIP_Bool             copyheuristics,     /**< should the heuristics be copied */
   SCIP_Bool             copyeventhdlrs,     /**< should the event handlers be copied */
   SCIP_Bool             copynodeselectors,  /**< should the node selectors be copied */
   SCIP_Bool             copybranchrules,    /**< should the branchrules be copied */
   SCIP_Bool             copydisplays,       /**< should the display columns be copied */
   SCIP_Bool             copydialogs,        /**< should the dialogs be copied */
   SCIP_Bool             copynlpis,          /**< should the NLPIs be copied */
   SCIP_Bool             passmessagehdlr,    /**< should the message handler be passed */
   SCIP_Bool*            valid               /**< pointer to store whether plugins, in particular all constraint
                                              *   handlers which do not need constraints were validly copied */
   );

/** create a problem by copying the problem data of the source SCIP
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *  @note Do not change the source SCIP environment during the copying process
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_FREE
 *
 *  @post After calling this method targetscip reaches one of the following stages depending on if and when the solution
 *        process was interrupted:
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note sourcescip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcopyProb(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_HASHMAP*         varmap,             /**< a hashmap to store the mapping of source variables corresponding
                                              *   target variables, or NULL */
   SCIP_HASHMAP*         consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL */
   SCIP_Bool             global,             /**< create a global or a local copy? */
   const char*           name                /**< problem name */
   );

/** create a problem by copying the original problem data of the source SCIP
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *  @note Do not change the source SCIP environment during the copying process
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_FREE
 *
 *  @post After calling this method targetscip reaches one of the following stages depending on if and when the solution
 *        process was interrupted:
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note sourcescip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcopyOrigProb(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_HASHMAP*         varmap,             /**< a hashmap to store the mapping of source variables corresponding
                                              *   target variables, or NULL */
   SCIP_HASHMAP*         consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL  */
   const char*           name                /**< problem name of target */
   );

/** returns copy of the source variable; if there already is a copy of the source variable in the variable hash map,
 *  it is just returned as target variable; elsewise a new variable will be created and added to the target SCIP; this
 *  created variable is added to the variable hash map and returned as target variable
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *  @note Do not change the source SCIP environment during the copying process
 *  @note if a new variable was created, this variable will be added to the target-SCIP, but it is not captured
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note targetscip stage does not get changed
 *
 *  @note sourcescip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetVarCopy(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_VAR*             sourcevar,          /**< source variable */
   SCIP_VAR**            targetvar,          /**< pointer to store the target variable */
   SCIP_HASHMAP*         varmap,             /**< a hashmap to store the mapping of source variables to the corresponding
                                              *   target variables, or NULL */
   SCIP_HASHMAP*         consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL */
   SCIP_Bool             global,             /**< should global or local bounds be used? */
   SCIP_Bool*            success             /**< pointer to store whether the copying was successful or not */
   );

/** copies all active variables from source-SCIP and adds these variable to the target-SCIP; the mapping between these
 *  variables are stored in the variable hashmap, target-SCIP has to be in problem creation stage, fixed and aggregated
 *  variables do not get copied
 *
 *  @note the variables are added to the target-SCIP but not captured
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *  @note Do not change the source SCIP environment during the copying process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note sourcescip stage does not get changed
 *
 *  @note targetscip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcopyVars(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_HASHMAP*         varmap,             /**< a hashmap to store the mapping of source variables  to the corresponding
                                              *   target variables, or NULL */
   SCIP_HASHMAP*         consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL */
   SCIP_Bool             global              /**< should global or local bounds be used? */
   );

/** copies all original variables from source-SCIP and adds these variable to the target-SCIP; the mapping between these
 *  variables are stored in the variable hashmap, target-SCIP has to be in problem creation stage, fixed and aggregated
 *  variables do not get copied
 *
 *  @note the variables are added to the target-SCIP but not captured
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *  @note Do not change the source SCIP environment during the copying process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note sourcescip stage does not get changed
 *
 *  @note targetscip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcopyOrigVars(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_HASHMAP*         varmap,             /**< a hashmap to store the mapping of source variables to the corresponding
                                              *   target variables, or NULL */
   SCIP_HASHMAP*         consmap             /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL */
   );

/** returns copy of the source constraint; if there already is a copy of the source constraint in the constraint hash
 *  map, it is just returned as target constraint; elsewise a new constraint will be created; this created constraint is
 *  added to the constraint hash map and returned as target constraint; the variable map is used to map the variables of
 *  the source SCIP to the variables of the target SCIP
 *
 *  @warning If a constraint is marked to be checked for feasibility but not to be enforced, a LP or pseudo solution may
 *           be declared feasible even if it violates this particular constraint.  This constellation should only be
 *           used, if no LP or pseudo solution can violate the constraint -- e.g. if a local constraint is redundant due
 *           to the variable's local bounds.
 *
 *  @note The constraint is not added to the target SCIP. You can check whether a constraint is added by calling
 *        SCIPconsIsAdded(). (If you mix SCIPgetConsCopy() with SCIPcopyConss() you should pay attention to what you add
 *        explicitly and what is already added.)
 *
 *  @note The constraint is always captured, either during the creation of the copy or after finding the copy of the
 *        constraint in the constraint hash map
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *  @note Do not change the source SCIP environment during the copying process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @note sourcescip stage does not get changed
 *
 *  @note targetscip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetConsCopy(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_CONS*            sourcecons,         /**< source constraint of the source SCIP */
   SCIP_CONS**           targetcons,         /**< pointer to store the created target constraint */
   SCIP_CONSHDLR*        sourceconshdlr,     /**< source constraint handler for this constraint */
   SCIP_HASHMAP*         varmap,             /**< a SCIP_HASHMAP mapping variables of the source SCIP to the corresponding
                                              *   variables of the target SCIP, or NULL */
   SCIP_HASHMAP*         consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL */
   const char*           name,               /**< name of constraint, or NULL if the name of the source constraint should be used */
   SCIP_Bool             initial,            /**< should the LP relaxation of constraint be in the initial LP? */
   SCIP_Bool             separate,           /**< should the constraint be separated during LP processing? */
   SCIP_Bool             enforce,            /**< should the constraint be enforced during node processing? */
   SCIP_Bool             check,              /**< should the constraint be checked for feasibility? */
   SCIP_Bool             propagate,          /**< should the constraint be propagated during node processing? */
   SCIP_Bool             local,              /**< is constraint only valid locally? */
   SCIP_Bool             modifiable,         /**< is constraint modifiable (subject to column generation)? */
   SCIP_Bool             dynamic,            /**< is constraint subject to aging? */
   SCIP_Bool             removable,          /**< should the relaxation be removed from the LP due to aging or cleanup? */
   SCIP_Bool             stickingatnode,     /**< should the constraint always be kept at the node where it was added, even
                                              *   if it may be moved to a more global node? */
   SCIP_Bool             global,             /**< create a global or a local copy? */
   SCIP_Bool*            success             /**< pointer to store whether the copying was successful or not */
   );

/** copies constraints from the source-SCIP and adds these to the target-SCIP; for mapping the
 *  variables between the source and the target SCIP a hash map can be given; if the variable hash
 *  map is NULL or necessary variable mapping is missing, the required variables are created in the
 *  target-SCIP and added to the hash map, if not NULL; all variables which are created are added to
 *  the target-SCIP but not (user) captured; if the constraint hash map is not NULL the mapping
 *  between the constraints of the source and target-SCIP is stored
 *
 *  @note the constraints are added to the target-SCIP but are not (user) captured in the target SCIP. (If you mix
 *        SCIPgetConsCopy() with SCIPcopyConss() you should pay attention to what you add explicitly and what is already
 *        added.) You can check whether a constraint is added by calling SCIPconsIsAdded().
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *  @note Do not change the source SCIP environment during the copying process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note sourcescip stage does not get changed
 *
 *  @note targetscip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcopyConss(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_HASHMAP*         varmap,             /**< a SCIP_HASHMAP mapping variables of the source SCIP to the corresponding
                                              *   variables of the target SCIP, must not be NULL! */
   SCIP_HASHMAP*         consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL */
   SCIP_Bool             global,             /**< create a global or a local copy? */
   SCIP_Bool             enablepricing,      /**< should pricing be enabled in copied SCIP instance? 
                                              *   If TRUE, the modifiable flag of constraints will be copied. */
   SCIP_Bool*            valid               /**< pointer to store whether all constraints were validly copied */
   );

/** copies all original constraints from the source-SCIP and adds these to the target-SCIP; for mapping the
 *  variables between the source and the target SCIP a hash map can be given; if the variable hash
 *  map is NULL or necessary variable mapping is missing, the required variables are created in the
 *  target-SCIP and added to the hash map, if not NULL; all variables which are created are added to
 *  the target-SCIP but not (user) captured; if the constraint hash map is not NULL the mapping
 *  between the constraints of the source and target-SCIP is stored
 *
 *  @note the constraints are added to the target-SCIP but are not (user) captured in the target SCIP. (If you mix
 *        SCIPgetConsCopy() with SCIPcopyConss() you should pay attention to what you add explicitly and what is already
 *        added.) You can check whether a constraint is added by calling SCIPconsIsAdded().
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *  @note Do not change the source SCIP environment during the copying process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note sourcescip stage does not get changed
 *
 *  @note targetscip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcopyOrigConss(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_HASHMAP*         varmap,             /**< a SCIP_HASHMAP mapping variables of the source SCIP to the corresponding
                                              *   variables of the target SCIP, or NULL */
   SCIP_HASHMAP*         consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL */
   SCIP_Bool             enablepricing,      /**< should pricing be enabled in copied SCIP instance?
                                              *   If TRUE, the modifiable flag of constraints will be copied. */
   SCIP_Bool*            valid               /**< pointer to store whether all constraints were validly copied */
   );

/** convert all active cuts from cutpool to linear constraints
 *
 *  @note Do not change the source SCIP environment during the copying process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @note SCIP stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPconvertCutsToConss(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HASHMAP*         varmap,             /**< a hashmap to store the mapping of source variables corresponding
                                              *   target variables, or NULL */
   SCIP_HASHMAP*         consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL */
   SCIP_Bool             global,             /**< create a global or a local copy? */
   int*                  ncutsadded          /**< pointer to store number of added cuts, or NULL */
   );

/** copies all active cuts from cutpool of sourcescip to linear constraints in targetscip
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *  @note Do not change the source SCIP environment during the copying process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @note sourcescip stage does not get changed
 *
 *  @note targetscip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcopyCuts(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_HASHMAP*         varmap,             /**< a hashmap to store the mapping of source variables corresponding
                                              *   target variables, or NULL */
   SCIP_HASHMAP*         consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL */
   SCIP_Bool             global,             /**< create a global or a local copy? */
   int*                  ncutsadded          /**< pointer to store number of copied cuts, or NULL */
   );

/** copies parameter settings from sourcescip to targetscip
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *  @note Do not change the source SCIP environment during the copying process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_FREE
 *
 *  @note sourcescip stage does not get changed
 *
 *  @note targetscip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcopyParamSettings(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip          /**< target SCIP data structure */
   );

/** gets depth of current scip instance (increased by each copy call)
 *
 *  @return Depth of subscip of SCIP is returned.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note SCIP stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetSubscipDepth(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** copies source SCIP to target SCIP; the copying process is done in the following order:
 *  1) copy the plugins
 *  2) copy the settings
 *  3) create problem data in target-SCIP and copy the problem data of the source-SCIP
 *  4) copy all active variables
 *  5) copy all constraints
 *
 *  @note all variables and constraints which are created in the target-SCIP are not (user) captured
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *        Also, 'passmessagehdlr' should be set to FALSE.
 *  @note Do not change the source SCIP environment during the copying process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_FREE
 *
 *  @note sourcescip stage does not get changed
 *
 *  @note targetscip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcopy(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_HASHMAP*         varmap,             /**< a hashmap to store the mapping of source variables corresponding
                                              *   target variables, or NULL */
   SCIP_HASHMAP*         consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL */
   const char*           suffix,             /**< suffix which will be added to the names of the source SCIP */
   SCIP_Bool             global,             /**< create a global or a local copy? */
   SCIP_Bool             enablepricing,      /**< should pricing be enabled in copied SCIP instance? If TRUE, pricer
                                              *   plugins will be copied and activated, and the modifiable flag of
                                              *   constraints will be respected. If FALSE, valid will be set to FALSE, when
                                              *   there are pricers present */
   SCIP_Bool             passmessagehdlr,    /**< should the message handler be passed */
   SCIP_Bool*            valid               /**< pointer to store whether the copying was valid or not */
   );

/** copies source SCIP original problem to target SCIP; the copying process is done in the following order:
 *  1) copy the plugins
 *  2) copy the settings
 *  3) create problem data in target-SCIP and copy the original problem data of the source-SCIP
 *  4) copy all original variables
 *  5) copy all original constraints
 *
 *  @note all variables and constraints which are created in the target-SCIP are not (user) captured
 *
 *  @note In a multi thread case, you need to lock the copying procedure from outside with a mutex.
 *        Also, 'passmessagehdlr' should be set to FALSE.
 *  @note Do not change the source SCIP environment during the copying process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if sourcescip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @pre This method can be called if targetscip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_FREE
 *
 *  @note sourcescip stage does not get changed
 *
 *  @note targetscip stage does not get changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcopyOrig(
   SCIP*                 sourcescip,         /**< source SCIP data structure */
   SCIP*                 targetscip,         /**< target SCIP data structure */
   SCIP_HASHMAP*         varmap,             /**< a hashmap to store the mapping of source variables corresponding
                                              *   target variables, or NULL */
   SCIP_HASHMAP*         consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints, or NULL */
   const char*           suffix,             /**< suffix which will be added to the names of the target SCIP, might be empty */
   SCIP_Bool             enablepricing,      /**< should pricing be enabled in copied SCIP instance? If TRUE, pricer
                                              *   plugins will be copied and activated, and the modifiable flag of
                                              *   constraints will be respected. If FALSE, valid will be set to FALSE, when
                                              *   there are pricers present */
   SCIP_Bool             passmessagehdlr,    /**< should the message handler be passed */
   SCIP_Bool*            valid               /**< pointer to store whether the copying was valid or not */
   );

/**@} */

/*
 * parameter settings
 */

/**@name Parameter Methods */
/**@{ */

/** creates a SCIP_Bool parameter, sets it to its default value, and adds it to the parameter set
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPaddBoolParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   SCIP_Bool*            valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   SCIP_Bool             defaultvalue,       /**< default value of the parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** creates a int parameter, sets it to its default value, and adds it to the parameter set
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPaddIntParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   int*                  valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   int                   defaultvalue,       /**< default value of the parameter */
   int                   minvalue,           /**< minimum value for parameter */
   int                   maxvalue,           /**< maximum value for parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** creates a SCIP_Longint parameter, sets it to its default value, and adds it to the parameter set
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPaddLongintParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   SCIP_Longint*         valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   SCIP_Longint          defaultvalue,       /**< default value of the parameter */
   SCIP_Longint          minvalue,           /**< minimum value for parameter */
   SCIP_Longint          maxvalue,           /**< maximum value for parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** creates a SCIP_Real parameter, sets it to its default value, and adds it to the parameter set
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPaddRealParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   SCIP_Real*            valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   SCIP_Real             defaultvalue,       /**< default value of the parameter */
   SCIP_Real             minvalue,           /**< minimum value for parameter */
   SCIP_Real             maxvalue,           /**< maximum value for parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** creates a char parameter, sets it to its default value, and adds it to the parameter set
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPaddCharParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   char*                 valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   char                  defaultvalue,       /**< default value of the parameter */
   const char*           allowedvalues,      /**< array with possible parameter values, or NULL if not restricted */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** creates a string(char*) parameter, sets it to its default value, and adds it to the parameter set
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPaddStringParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   char**                valueptr,           /**< pointer to store the current parameter value, or NULL; if not NULL then *valueptr should be NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   const char*           defaultvalue,       /**< default value of the parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** gets the fixing status of an existing parameter
 *
 *  @return TRUE if the parameter is fixed to a value, otherwise FALSE.
 */
EXTERN
SCIP_Bool SCIPisParamFixed(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of the parameter */
   );

/** returns the pointer to the SCIP parameter with the given name
 *
 *  @return pointer to the parameter with the given name
 */
EXTERN
SCIP_PARAM* SCIPgetParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of the parameter */
   );

/** gets the value of an existing SCIP_Bool parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPgetBoolParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Bool*            value               /**< pointer to store the parameter */
   );

/** gets the value of an existing int parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPgetIntParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   int*                  value               /**< pointer to store the parameter */
   );

/** gets the value of an existing SCIP_Longint parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPgetLongintParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Longint*         value               /**< pointer to store the parameter */
   );

/** gets the value of an existing SCIP_Real parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPgetRealParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Real*            value               /**< pointer to store the parameter */
   );

/** gets the value of an existing char parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPgetCharParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   char*                 value               /**< pointer to store the parameter */
   );

/** gets the value of an existing string(char*) parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPgetStringParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   char**                value               /**< pointer to store the parameter */
   );

/** fixes the value of an existing parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @note: Be careful with this method! Some general settings, e.g., the time or node limit, should not be fixed because
 *         they have to be changed for sub-SCIPs.
 */
EXTERN
SCIP_RETCODE SCIPfixParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of the parameter */
   );

/** unfixes the value of an existing parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPunfixParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of the parameter */
   );

/** changes the value of an existing parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   void*                 value               /**< new value of the parameter */
   );

/** changes the value of an existing SCIP_Bool parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPchgBoolParam(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAM*           param,              /**< parameter */
   SCIP_Bool             value               /**< new value of the parameter */
   );

/** changes the value of an existing SCIP_Bool parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetBoolParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Bool             value               /**< new value of the parameter */
   );

/** changes the value of an existing int parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPchgIntParam(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAM*           param,              /**< parameter */
   int                   value               /**< new value of the parameter */
   );

/** changes the value of an existing int parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetIntParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   int                   value               /**< new value of the parameter */
   );

/** changes the value of an existing SCIP_Longint parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPchgLongintParam(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAM*           param,              /**< parameter */
   SCIP_Longint          value               /**< new value of the parameter */
   );

/** changes the value of an existing SCIP_Longint parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetLongintParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Longint          value               /**< new value of the parameter */
   );

/** changes the value of an existing SCIP_Real parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPchgRealParam(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAM*           param,              /**< parameter */
   SCIP_Real             value               /**< new value of the parameter */
   );

/** changes the value of an existing SCIP_Real parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetRealParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Real             value               /**< new value of the parameter */
   );

/** changes the value of an existing char parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPchgCharParam(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAM*           param,              /**< parameter */
   char                  value               /**< new value of the parameter */
   );

/** changes the value of an existing char parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetCharParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   char                  value               /**< new value of the parameter */
   );

/** changes the value of an existing string(char*) parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPchgStringParam(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAM*           param,              /**< parameter */
   const char*           value               /**< new value of the parameter */
   );

/** changes the value of an existing string(char*) parameter
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetStringParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           value               /**< new value of the parameter */
   );

/** reads parameters from a file
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPreadParams(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename            /**< file name */
   );

/** writes all parameters in the parameter set to a file
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPwriteParams(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< file name, or NULL for stdout */
   SCIP_Bool             comments,           /**< should parameter descriptions be written as comments? */
   SCIP_Bool             onlychanged         /**< should only the parameters been written, that are changed from default? */
   );

/** resets a single parameter to its default value
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPresetParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of the parameter */
   );

/** resets all parameters to their default values
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPresetParams(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets parameters to
 *
 *  - SCIP_PARAMEMPHASIS_DEFAULT to use default values (see also SCIPresetParams())
 *  - SCIP_PARAMEMPHASIS_COUNTER to get feasible and "fast" counting process
 *  - SCIP_PARAMEMPHASIS_CPSOLVER to get CP like search (e.g. no LP relaxation)
 *  - SCIP_PARAMEMPHASIS_EASYCIP to solve easy problems fast
 *  - SCIP_PARAMEMPHASIS_FEASIBILITY to detect feasibility fast
 *  - SCIP_PARAMEMPHASIS_HARDLP to be capable to handle hard LPs
 *  - SCIP_PARAMEMPHASIS_OPTIMALITY to prove optimality fast
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetEmphasis(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAMEMPHASIS    paramemphasis,      /**< parameter settings */
   SCIP_Bool             quiet               /**< should the parameter be set quiet (no output) */
   );

/** sets parameters to deactivate separators and heuristics that use auxiliary SCIP instances; should be called for
 *  auxiliary SCIP instances to avoid recursion
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetSubscipsOff(
   SCIP*                 scip,               /**< (auxiliary) SCIP data structure */
   SCIP_Bool             quiet               /**< should the parameter be set quiet (no output) */
   );

/** sets heuristic parameters values to
 *
 *  - SCIP_PARAMSETTING_DEFAULT which are the default values of all heuristic parameters
 *  - SCIP_PARAMSETTING_FAST such that the time spend for heuristic is decreased
 *  - SCIP_PARAMSETTING_AGGRESSIVE such that the heuristic are called more aggregative
 *  - SCIP_PARAMSETTING_OFF which turn off all heuristics
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetHeuristics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAMSETTING     paramsetting,       /**< parameter settings */
   SCIP_Bool             quiet               /**< should the parameter be set quiet (no output) */
   );

/** sets presolving parameters to
 *
 *  - SCIP_PARAMSETTING_DEFAULT which are the default values of all presolving parameters
 *  - SCIP_PARAMSETTING_FAST such that the time spend for presolving is decreased
 *  - SCIP_PARAMSETTING_AGGRESSIVE such that the presolving is more aggregative
 *  - SCIP_PARAMSETTING_OFF which turn off all presolving
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetPresolving(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAMSETTING     paramsetting,       /**< parameter settings */
   SCIP_Bool             quiet               /**< should the parameter be set quiet (no output) */
   );

/** sets separating parameters to
 *
 *  - SCIP_PARAMSETTING_DEFAULT which are the default values of all separating parameters
 *  - SCIP_PARAMSETTING_FAST such that the time spend for separating is decreased
 *  - SCIP_PARAMSETTING_AGGRESSIVE such that the separating is done more aggregative
 *  - SCIP_PARAMSETTING_OFF which turn off all separating
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetSeparating(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAMSETTING     paramsetting,       /**< parameter settings */
   SCIP_Bool             quiet               /**< should the parameter be set quiet (no output) */
   );

/** returns the array of all available SCIP parameters
 *
 *  @return SCIP_PARAM* array, containing all SCIP parameters.
 */
EXTERN
SCIP_PARAM** SCIPgetParams(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the total number of all available SCIP parameters
 *
 *  @return number of all SCIP parameters.
 */
EXTERN
int SCIPgetNParams(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@} */


/*
 * SCIP user functionality methods: managing plugins
 */

/**@name SCIP User Functionality Methods: Managing Plugins */
/**@{ */

/** creates a reader and includes it in SCIP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note method has all reader callbacks as arguments and is thus changed every time a new callback is added
 *        in future releases; consider using SCIPincludeReaderBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludeReader(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of reader */
   const char*           desc,               /**< description of reader */
   const char*           extension,          /**< file extension that reader processes */
   SCIP_DECL_READERCOPY  ((*readercopy)),    /**< copy method of reader or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_READERFREE  ((*readerfree)),    /**< destructor of reader */
   SCIP_DECL_READERREAD  ((*readerread)),    /**< read method */
   SCIP_DECL_READERWRITE ((*readerwrite)),   /**< write method */
   SCIP_READERDATA*      readerdata          /**< reader data */
   );

/** creates a reader and includes it in SCIP. All non-fundamental (or optional) callbacks will be set to NULL.
 *  Optional callbacks can be set via specific setter functions, see
 *  SCIPsetReaderCopy(), SCIPsetReaderFree(), SCIPsetReaderRead(), SCIPsetReaderWrite().
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note if you want to set all callbacks with a single method call, consider using SCIPincludeReader() instead
 */
EXTERN
SCIP_RETCODE SCIPincludeReaderBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER**         readerptr,          /**< reference to reader pointer, or NULL */
   const char*           name,               /**< name of reader */
   const char*           desc,               /**< description of reader */
   const char*           extension,          /**< file extension that reader processes */
   SCIP_READERDATA*      readerdata          /**< reader data */
   );

/** set copy method of reader
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetReaderCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER*          reader,             /**< reader */
   SCIP_DECL_READERCOPY  ((*readercopy))     /**< copy method of reader or NULL if you don't want to copy your plugin into sub-SCIPs */
   );

/** set deinitialization method of reader
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetReaderFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER*          reader,             /**< reader */
   SCIP_DECL_READERFREE  ((*readerfree))     /**< destructor of reader */
   );

/** set read method of reader
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetReaderRead(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER*          reader,             /**< reader */
   SCIP_DECL_READERREAD  ((*readerread))     /**< read method of reader */
   );

/** set write method of reader
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetReaderWrite(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER*          reader,             /**< reader */
   SCIP_DECL_READERWRITE ((*readerwrite))    /**< write method of reader */
   );

/** returns the reader of the given name, or NULL if not existing */
EXTERN
SCIP_READER* SCIPfindReader(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of reader */
   );

/** returns the array of currently available readers */
EXTERN
SCIP_READER** SCIPgetReaders(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available readers */
EXTERN
int SCIPgetNReaders(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** creates a variable pricer and includes it in SCIP
 *  To use the variable pricer for solving a problem, it first has to be activated with a call to SCIPactivatePricer().
 *  This should be done during the problem creation stage.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note method has all pricer callbacks as arguments and is thus changed every time a new callback is added
 *        in future releases; consider using SCIPincludePricerBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludePricer(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of variable pricer */
   const char*           desc,               /**< description of variable pricer */
   int                   priority,           /**< priority of the variable pricer */
   SCIP_Bool             delay,              /**< should the pricer be delayed until no other pricers or already existing
                                              *   problem variables with negative reduced costs are found?
                                              *   if this is set to FALSE it may happen that the pricer produces columns
                                              *   that already exist in the problem (which are also priced in by the
                                              *   default problem variable pricing in the same round) */
   SCIP_DECL_PRICERCOPY  ((*pricercopy)),    /**< copy method of variable pricer or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_PRICERFREE  ((*pricerfree)),    /**< destructor of variable pricer */
   SCIP_DECL_PRICERINIT  ((*pricerinit)),    /**< initialize variable pricer */
   SCIP_DECL_PRICEREXIT  ((*pricerexit)),    /**< deinitialize variable pricer */
   SCIP_DECL_PRICERINITSOL((*pricerinitsol)),/**< solving process initialization method of variable pricer */
   SCIP_DECL_PRICEREXITSOL((*pricerexitsol)),/**< solving process deinitialization method of variable pricer */
   SCIP_DECL_PRICERREDCOST((*pricerredcost)),/**< reduced cost pricing method of variable pricer for feasible LPs */
   SCIP_DECL_PRICERFARKAS((*pricerfarkas)),  /**< Farkas pricing method of variable pricer for infeasible LPs */
   SCIP_PRICERDATA*      pricerdata          /**< variable pricer data */
   );

/** creates a variable pricer and includes it in SCIP with all non-fundamental callbacks set to NULL;
 *  if needed, these can be added afterwards via setter functions SCIPsetPricerCopy(), SCIPsetPricerFree(),
 *  SCIPsetPricerInity(), SCIPsetPricerExit(), SCIPsetPricerInitsol(), SCIPsetPricerExitsol(),
 *  SCIPsetPricerFarkas();
 *
 *  To use the variable pricer for solving a problem, it first has to be activated with a call to SCIPactivatePricer().
 *  This should be done during the problem creation stage.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note if you want to set all callbacks with a single method call, consider using SCIPincludePricer() instead
 */
EXTERN
SCIP_RETCODE SCIPincludePricerBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER**         pricerptr,          /**< reference to a pricer, or NULL */
   const char*           name,               /**< name of variable pricer */
   const char*           desc,               /**< description of variable pricer */
   int                   priority,           /**< priority of the variable pricer */
   SCIP_Bool             delay,              /**< should the pricer be delayed until no other pricers or already existing
                                              *   problem variables with negative reduced costs are found?
                                              *   if this is set to FALSE it may happen that the pricer produces columns
                                              *   that already exist in the problem (which are also priced in by the
                                              *   default problem variable pricing in the same round) */
   SCIP_DECL_PRICERREDCOST((*pricerredcost)),/**< reduced cost pricing method of variable pricer for feasible LPs */
   SCIP_DECL_PRICERFARKAS((*pricerfarkas)),  /**< Farkas pricing method of variable pricer for infeasible LPs */
   SCIP_PRICERDATA*      pricerdata          /**< variable pricer data */
   );

/** sets copy method of pricer
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetPricerCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer,             /**< pricer */
   SCIP_DECL_PRICERCOPY ((*pricercopy))     /**< copy method of pricer or NULL if you don't want to copy your plugin into sub-SCIPs */
   );

/** sets destructor method of pricer
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetPricerFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer,             /**< pricer */
   SCIP_DECL_PRICERFREE ((*pricerfree))      /**< destructor of pricer */
   );

/** sets initialization method of pricer
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetPricerInit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer,             /**< pricer */
   SCIP_DECL_PRICERINIT  ((*pricerinit))     /**< initialize pricer */
   );

/** sets deinitialization method of pricer
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetPricerExit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer,             /**< pricer */
   SCIP_DECL_PRICEREXIT  ((*pricerexit))     /**< deinitialize pricer */
   );

/** sets solving process initialization method of pricer
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetPricerInitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer,             /**< pricer */
   SCIP_DECL_PRICERINITSOL ((*pricerinitsol))/**< solving process initialization method of pricer */
   );

/** sets solving process deinitialization method of pricer
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetPricerExitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer,             /**< pricer */
   SCIP_DECL_PRICEREXITSOL((*pricerexitsol)) /**< solving process deinitialization method of pricer */
   );

/** returns the variable pricer of the given name, or NULL if not existing */
EXTERN
SCIP_PRICER* SCIPfindPricer(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of variable pricer */
   );

/** returns the array of currently available variable pricers; active pricers are in the first slots of the array */
EXTERN
SCIP_PRICER** SCIPgetPricers(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available variable pricers */
EXTERN
int SCIPgetNPricers(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently active variable pricers, that are used in the LP solving loop */
EXTERN
int SCIPgetNActivePricers(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the priority of a variable pricer */
EXTERN
SCIP_RETCODE SCIPsetPricerPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer,             /**< variable pricer */
   int                   priority            /**< new priority of the variable pricer */
   );

/** activates pricer to be used for the current problem
 *  This method should be called during the problem creation stage for all pricers that are necessary to solve
 *  the problem model.
 *  The pricers are automatically deactivated when the problem is freed.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPactivatePricer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer              /**< variable pricer */
   );

/** deactivates pricer
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPdeactivatePricer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer              /**< variable pricer */
   );

/** creates a constraint handler and includes it in SCIP.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note method has all constraint handler callbacks as arguments and is thus changed every time a new
 *        callback is added
 *        in future releases; consider using SCIPincludeConshdlrBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludeConshdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of constraint handler */
   const char*           desc,               /**< description of constraint handler */
   int                   sepapriority,       /**< priority of the constraint handler for separation */
   int                   enfopriority,       /**< priority of the constraint handler for constraint enforcing */
   int                   chckpriority,       /**< priority of the constraint handler for checking feasibility (and propagation) */
   int                   sepafreq,           /**< frequency for separating cuts; zero means to separate only in the root node */
   int                   propfreq,           /**< frequency for propagating domains; zero means only preprocessing propagation */
   int                   eagerfreq,          /**< frequency for using all instead of only the useful constraints in separation,
                                              *   propagation and enforcement, -1 for no eager evaluations, 0 for first only */
   int                   maxprerounds,       /**< maximal number of presolving rounds the constraint handler participates in (-1: no limit) */
   SCIP_Bool             delaysepa,          /**< should separation method be delayed, if other separators found cuts? */
   SCIP_Bool             delayprop,          /**< should propagation method be delayed, if other propagators found reductions? */
   SCIP_Bool             delaypresol,        /**< should presolving method be delayed, if other presolvers found reductions? */
   SCIP_Bool             needscons,          /**< should the constraint handler be skipped, if no constraints are available? */
   SCIP_PROPTIMING       timingmask,         /**< positions in the node solving loop where propagators should be executed */
   SCIP_DECL_CONSHDLRCOPY((*conshdlrcopy)),  /**< copy method of constraint handler or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_CONSFREE    ((*consfree)),      /**< destructor of constraint handler */
   SCIP_DECL_CONSINIT    ((*consinit)),      /**< initialize constraint handler */
   SCIP_DECL_CONSEXIT    ((*consexit)),      /**< deinitialize constraint handler */
   SCIP_DECL_CONSINITPRE ((*consinitpre)),   /**< presolving initialization method of constraint handler */
   SCIP_DECL_CONSEXITPRE ((*consexitpre)),   /**< presolving deinitialization method of constraint handler */
   SCIP_DECL_CONSINITSOL ((*consinitsol)),   /**< solving process initialization method of constraint handler */
   SCIP_DECL_CONSEXITSOL ((*consexitsol)),   /**< solving process deinitialization method of constraint handler */
   SCIP_DECL_CONSDELETE  ((*consdelete)),    /**< free specific constraint data */
   SCIP_DECL_CONSTRANS   ((*constrans)),     /**< transform constraint data into data belonging to the transformed problem */
   SCIP_DECL_CONSINITLP  ((*consinitlp)),    /**< initialize LP with relaxations of "initial" constraints */
   SCIP_DECL_CONSSEPALP  ((*conssepalp)),    /**< separate cutting planes for LP solution */
   SCIP_DECL_CONSSEPASOL ((*conssepasol)),   /**< separate cutting planes for arbitrary primal solution */
   SCIP_DECL_CONSENFOLP  ((*consenfolp)),    /**< enforcing constraints for LP solutions */
   SCIP_DECL_CONSENFOPS  ((*consenfops)),    /**< enforcing constraints for pseudo solutions */
   SCIP_DECL_CONSCHECK   ((*conscheck)),     /**< check feasibility of primal solution */
   SCIP_DECL_CONSPROP    ((*consprop)),      /**< propagate variable domains */
   SCIP_DECL_CONSPRESOL  ((*conspresol)),    /**< presolving method */
   SCIP_DECL_CONSRESPROP ((*consresprop)),   /**< propagation conflict resolving method */
   SCIP_DECL_CONSLOCK    ((*conslock)),      /**< variable rounding lock method */
   SCIP_DECL_CONSACTIVE  ((*consactive)),    /**< activation notification method */
   SCIP_DECL_CONSDEACTIVE((*consdeactive)),  /**< deactivation notification method */
   SCIP_DECL_CONSENABLE  ((*consenable)),    /**< enabling notification method */
   SCIP_DECL_CONSDISABLE ((*consdisable)),   /**< disabling notification method */
   SCIP_DECL_CONSDELVARS ((*consdelvars)),   /**< variable deletion method */
   SCIP_DECL_CONSPRINT   ((*consprint)),     /**< constraint display method */
   SCIP_DECL_CONSCOPY    ((*conscopy)),      /**< constraint copying method */
   SCIP_DECL_CONSPARSE   ((*consparse)),     /**< constraint parsing method */
   SCIP_DECL_CONSGETVARS ((*consgetvars)),   /**< constraint get variables method */
   SCIP_DECL_CONSGETNVARS((*consgetnvars)),  /**< constraint get number of variable method */
   SCIP_CONSHDLRDATA*    conshdlrdata        /**< constraint handler data */
   );

/** creates a constraint handler and includes it in SCIP. All non-fundamental (or optional) callbacks will be set to NULL.
 *  Optional callbacks can be set via specific setter functions, see SCIPsetConshdlrInit(), SCIPsetConshdlrExit(),
 *  SCIPsetConshdlrCopy(), SCIPsetConshdlrFree(), SCIPsetConshdlrInitsol(), SCIPsetConshdlrExitsol(),
 *  SCIPsetConshdlrInitpre(), SCIPsetConshdlrExitpre(), SCIPsetConshdlrPresol(), SCIPsetConshdlrDelete(),
 *  SCIPsetConshdlrDelvars(), SCIPsetConshdlrInitlp(), SCIPsetConshdlrActive(), SCIPsetConshdlrDeactive(),
 *  SCIPsetConshdlrEnable(), SCIPsetConshdlrDisable(), SCIPsetConshdlrResprop(), SCIPsetConshdlrTrans(),
 *  SCIPsetConshdlrPrint(), and SCIPsetConshdlrParse().
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *
 *  @note if you want to set all callbacks with a single method call, consider using SCIPincludeConshdlr() instead
 */
EXTERN
SCIP_RETCODE SCIPincludeConshdlrBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR**       conshdlrptr,        /**< reference to a constraint handler pointer, or NULL */
   const char*           name,               /**< name of constraint handler */
   const char*           desc,               /**< description of constraint handler */
   int                   enfopriority,       /**< priority of the constraint handler for constraint enforcing */
   int                   chckpriority,       /**< priority of the constraint handler for checking feasibility (and propagation) */
   int                   eagerfreq,          /**< frequency for using all instead of only the useful constraints in separation,
                                              *   propagation and enforcement, -1 for no eager evaluations, 0 for first only */
   SCIP_Bool             needscons,          /**< should the constraint handler be skipped, if no constraints are available? */
   SCIP_DECL_CONSENFOLP  ((*consenfolp)),    /**< enforcing constraints for LP solutions */
   SCIP_DECL_CONSENFOPS  ((*consenfops)),    /**< enforcing constraints for pseudo solutions */
   SCIP_DECL_CONSCHECK   ((*conscheck)),     /**< check feasibility of primal solution */
   SCIP_DECL_CONSLOCK    ((*conslock)),      /**< variable rounding lock method */
   SCIP_CONSHDLRDATA*    conshdlrdata        /**< constraint handler data */
   );

/** sets all separation related callbacks/parameters of the constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrSepa(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSSEPALP  ((*conssepalp)),    /**< separate cutting planes for LP solution */
   SCIP_DECL_CONSSEPASOL ((*conssepasol)),   /**< separate cutting planes for arbitrary primal solution */
   int                   sepafreq,           /**< frequency for separating cuts; zero means to separate only in the root node */
   int                   sepapriority,       /**< priority of the constraint handler for separation */
   SCIP_Bool             delaysepa           /**< should separation method be delayed, if other separators found cuts? */
   );

/** sets both the propagation callback and the propagation frequency of the constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrProp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSPROP    ((*consprop)),      /**< propagate variable domains */
   int                   propfreq,           /**< frequency for propagating domains; zero means only preprocessing propagation */
   SCIP_Bool             delayprop,          /**< should propagation method be delayed, if other propagators found reductions? */
   SCIP_PROPTIMING       timingmask          /**< positions in the node solving loop where propagators should be executed */
   );

/** sets copy method of both the constraint handler and each associated constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSHDLRCOPY((*conshdlrcopy)),  /**< copy method of constraint handler or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_CONSCOPY    ((*conscopy))       /**< constraint copying method */
   );

/** sets destructor method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSFREE    ((*consfree))       /**< destructor of constraint handler */
   );

/** sets initialization method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrInit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSINIT    ((*consinit))   /**< initialize constraint handler */
   );

/** sets deinitialization method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrExit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSEXIT    ((*consexit))       /**< deinitialize constraint handler */
   );

/** sets solving process initialization method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrInitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSINITSOL((*consinitsol))     /**< solving process initialization method of constraint handler */
   );

/** sets solving process deinitialization method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrExitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSEXITSOL ((*consexitsol))/**< solving process deinitialization method of constraint handler */
   );

/** sets preprocessing initialization method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrInitpre(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSINITPRE((*consinitpre))     /**< preprocessing initialization method of constraint handler */
   );

/** sets preprocessing deinitialization method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrExitpre(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSEXITPRE((*consexitpre))     /**< preprocessing deinitialization method of constraint handler */
   );

/** sets presolving method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrPresol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSPRESOL  ((*conspresol)),    /**< presolving method of constraint handler */
   int                   maxprerounds,       /**< maximal number of presolving rounds the constraint handler participates in (-1: no limit) */
   SCIP_Bool             delaypresol         /**< should presolving method be delayed, if other presolvers found reductions? */
   );

/** sets method of constraint handler to free specific constraint data
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrDelete(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSDELETE  ((*consdelete))     /**< free specific constraint data */
   );

/** sets method of constraint handler to transform constraint data into data belonging to the transformed problem
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrTrans(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSTRANS   ((*constrans))      /**< transform constraint data into data belonging to the transformed problem */
   );

/** sets method of constraint handler to initialize LP with relaxations of "initial" constraints
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrInitlp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSINITLP  ((*consinitlp))     /**< initialize LP with relaxations of "initial" constraints */
   );

/** sets propagation conflict resolving method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrResprop(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSRESPROP ((*consresprop))    /**< propagation conflict resolving method */
   );

/** sets activation notification method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrActive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSACTIVE  ((*consactive))     /**< activation notification method */
   );

/** sets deactivation notification method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrDeactive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSDEACTIVE((*consdeactive))   /**< deactivation notification method */
   );

/** sets enabling notification method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrEnable(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSENABLE  ((*consenable))     /**< enabling notification method */
   );

/** sets disabling notification method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrDisable(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSDISABLE ((*consdisable))    /**< disabling notification method */
   );

/** sets variable deletion method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrDelvars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSDELVARS ((*consdelvars))    /**< variable deletion method */
   );

/** sets constraint display method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrPrint(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSPRINT   ((*consprint))      /**< constraint display method */
   );

/** sets constraint parsing method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrParse(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSPARSE   ((*consparse))      /**< constraint parsing method */
   );

/** sets constraint variable getter method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrGetVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSGETVARS ((*consgetvars))    /**< constraint variable getter method */
   );

/** sets constraint variable number getter method of constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetConshdlrGetNVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_DECL_CONSGETNVARS((*consgetnvars))   /**< constraint variable number getter method */
   );

/** returns the constraint handler of the given name, or NULL if not existing */
EXTERN
SCIP_CONSHDLR* SCIPfindConshdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of constraint handler */
   );

/** returns the array of currently available constraint handlers */
EXTERN
SCIP_CONSHDLR** SCIPgetConshdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available constraint handlers */
EXTERN
int SCIPgetNConshdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** creates a conflict handler and includes it in SCIP
 *
 *  @note method has all conflict handler callbacks as arguments and is thus changed every time a new
 *        callback is added
 *        in future releases; consider using SCIPincludeConflicthdlrBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludeConflicthdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of conflict handler */
   const char*           desc,               /**< description of conflict handler */
   int                   priority,           /**< priority of the conflict handler */
   SCIP_DECL_CONFLICTCOPY((*conflictcopy)),  /**< copy method of conflict handler or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_CONFLICTFREE((*conflictfree)),  /**< destructor of conflict handler */
   SCIP_DECL_CONFLICTINIT((*conflictinit)),  /**< initialize conflict handler */
   SCIP_DECL_CONFLICTEXIT((*conflictexit)),  /**< deinitialize conflict handler */
   SCIP_DECL_CONFLICTINITSOL((*conflictinitsol)),/**< solving process initialization method of conflict handler */
   SCIP_DECL_CONFLICTEXITSOL((*conflictexitsol)),/**< solving process deinitialization method of conflict handler */
   SCIP_DECL_CONFLICTEXEC((*conflictexec)),  /**< conflict processing method of conflict handler */
   SCIP_CONFLICTHDLRDATA* conflicthdlrdata   /**< conflict handler data */
   );

/** creates a conflict handler and includes it in SCIP with its most fundamental callbacks. All non-fundamental
 *  (or optional) callbacks as, e.g., init and exit callbacks, will be set to NULL.
 *  Optional callbacks can be set via specific setter functions SCIPsetConflicthdlrCopy(), SCIPsetConflicthdlrFree(),
 *  SCIPsetConflicthdlrInit(), SCIPsetConflicthdlrExit(), SCIPsetConflicthdlrInitsol(),
 *  and SCIPsetConflicthdlrExitsol()
 *
 *  @note if you want to set all callbacks with a single method call, consider using SCIPincludeConflicthdlr() instead
 */
EXTERN
SCIP_RETCODE SCIPincludeConflicthdlrBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONFLICTHDLR**   conflicthdlrptr,    /**< reference to a conflict handler pointer, or NULL */
   const char*           name,               /**< name of conflict handler */
   const char*           desc,               /**< description of conflict handler */
   int                   priority,           /**< priority of the conflict handler */
   SCIP_DECL_CONFLICTEXEC((*conflictexec)),  /**< conflict processing method of conflict handler */
   SCIP_CONFLICTHDLRDATA* conflicthdlrdata   /**< conflict handler data */
   );

/** set copy method of conflict handler */
EXTERN
SCIP_RETCODE SCIPsetConflicthdlrCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONFLICTHDLR*    conflicthdlr,       /**< conflict handler */
   SCIP_DECL_CONFLICTCOPY((*conflictcopy))   /**< copy method of conflict handler */
   );

/** set destructor of conflict handler */
EXTERN
SCIP_RETCODE SCIPsetConflicthdlrFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONFLICTHDLR*    conflicthdlr,       /**< conflict handler */
   SCIP_DECL_CONFLICTFREE((*conflictfree))   /**< destructor of conflict handler */
   );

/** set initialization method of conflict handler */
EXTERN
SCIP_RETCODE SCIPsetConflicthdlrInit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONFLICTHDLR*    conflicthdlr,       /**< conflict handler */
   SCIP_DECL_CONFLICTINIT((*conflictinit))   /**< initialize conflict handler */
   );

/** set deinitialization method of conflict handler */
EXTERN
SCIP_RETCODE SCIPsetConflicthdlrExit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONFLICTHDLR*    conflicthdlr,       /**< conflict handler */
   SCIP_DECL_CONFLICTEXIT((*conflictexit))   /**< deinitialize conflict handler */
   );

/** set solving process initialization method of conflict handler */
EXTERN
SCIP_RETCODE SCIPsetConflicthdlrInitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONFLICTHDLR*    conflicthdlr,       /**< conflict handler */
   SCIP_DECL_CONFLICTINITSOL((*conflictinitsol))/**< solving process initialization method of conflict handler */
   );

/** set solving process deinitialization method of conflict handler */
EXTERN
SCIP_RETCODE SCIPsetConflicthdlrExitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONFLICTHDLR*    conflicthdlr,       /**< conflict handler */
   SCIP_DECL_CONFLICTEXITSOL((*conflictexitsol))/**< solving process deinitialization method of conflict handler */
   );

/** returns the conflict handler of the given name, or NULL if not existing */
EXTERN
SCIP_CONFLICTHDLR* SCIPfindConflicthdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of conflict handler */
   );

/** returns the array of currently available conflict handlers */
EXTERN
SCIP_CONFLICTHDLR** SCIPgetConflicthdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available conflict handlers */
EXTERN
int SCIPgetNConflicthdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the priority of a conflict handler */
EXTERN
SCIP_RETCODE SCIPsetConflicthdlrPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONFLICTHDLR*    conflicthdlr,       /**< conflict handler */
   int                   priority            /**< new priority of the conflict handler */
   );

/** creates a presolver and includes it in SCIP
 *
 *  @note method has all presolver callbacks as arguments and is thus changed every time a new
 *        callback is added
 *        in future releases; consider using SCIPincludePresolBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludePresol(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of presolver */
   const char*           desc,               /**< description of presolver */
   int                   priority,           /**< priority of the presolver (>= 0: before, < 0: after constraint handlers) */
   int                   maxrounds,          /**< maximal number of presolving rounds the presolver participates in (-1: no limit) */
   SCIP_Bool             delay,              /**< should presolver be delayed, if other presolvers found reductions? */
   SCIP_DECL_PRESOLCOPY  ((*presolcopy)),    /**< copy method of presolver or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_PRESOLFREE  ((*presolfree)),    /**< destructor of presolver to free user data (called when SCIP is exiting) */
   SCIP_DECL_PRESOLINIT  ((*presolinit)),    /**< initialization method of presolver (called after problem was transformed) */
   SCIP_DECL_PRESOLEXIT  ((*presolexit)),    /**< deinitialization method of presolver (called before transformed problem is freed) */
   SCIP_DECL_PRESOLINITPRE((*presolinitpre)),/**< presolving initialization method of presolver (called when presolving is about to begin) */
   SCIP_DECL_PRESOLEXITPRE((*presolexitpre)),/**< presolving deinitialization method of presolver (called after presolving has been finished) */
   SCIP_DECL_PRESOLEXEC  ((*presolexec)),    /**< execution method of presolver */
   SCIP_PRESOLDATA*      presoldata          /**< presolver data */
   );

/** Creates a presolver and includes it in SCIP with its fundamental callback. All non-fundamental (or optional)
 *  callbacks as, e.g., init and exit callbacks, will be set to NULL. Optional callbacks can be set via specific setter
 *  functions. These are SCIPsetPresolCopy(), SCIPsetPresolFree(), SCIPsetPresolInit(), SCIPsetPresolExit(),
 *  SCIPsetPresolInitpre(), and SCIPsetPresolExitPre().
 *
 *  @note if you want to set all callbacks with a single method call, consider using SCIPincludePresol() instead
 */
EXTERN
SCIP_RETCODE SCIPincludePresolBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOL**         presolptr,          /**< reference to presolver, or NULL */
   const char*           name,               /**< name of presolver */
   const char*           desc,               /**< description of presolver */
   int                   priority,           /**< priority of the presolver (>= 0: before, < 0: after constraint handlers) */
   int                   maxrounds,          /**< maximal number of presolving rounds the presolver participates in (-1: no limit) */
   SCIP_Bool             delay,              /**< should presolver be delayed, if other presolvers found reductions? */
   SCIP_DECL_PRESOLEXEC  ((*presolexec)),    /**< execution method of presolver */
   SCIP_PRESOLDATA*      presoldata          /**< presolver data */
   );

/** sets copy method of presolver */
EXTERN
SCIP_RETCODE SCIPsetPresolCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOL*          presol,             /**< presolver */
   SCIP_DECL_PRESOLCOPY  ((*presolcopy))     /**< copy method of presolver or NULL if you don't want to copy your plugin into sub-SCIPs */
   );

/** sets destructor method of presolver */
EXTERN
SCIP_RETCODE SCIPsetPresolFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOL*          presol,             /**< presolver */
   SCIP_DECL_PRESOLFREE ((*presolfree))      /**< destructor of presolver */
   );

/** sets initialization method of presolver */
EXTERN
SCIP_RETCODE SCIPsetPresolInit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOL*          presol,             /**< presolver */
   SCIP_DECL_PRESOLINIT ((*presolinit))      /**< initialize presolver */
   );

/** sets deinitialization method of presolver */
EXTERN
SCIP_RETCODE SCIPsetPresolExit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOL*          presol,             /**< presolver */
   SCIP_DECL_PRESOLEXIT  ((*presolexit))     /**< deinitialize presolver */
   );

/** sets solving process initialization method of presolver */
EXTERN
SCIP_RETCODE SCIPsetPresolInitpre(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOL*          presol,             /**< presolver */
   SCIP_DECL_PRESOLINITPRE ((*presolinitpre))/**< solving process initialization method of presolver */
   );

/** sets solving process deinitialization method of presolver */
SCIP_RETCODE SCIPsetPresolExitpre(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOL*          presol,             /**< presolver */
   SCIP_DECL_PRESOLEXITPRE ((*presolexitpre))/**< solving process deinitialization method of presolver */
   );

/** returns the presolver of the given name, or NULL if not existing */
EXTERN
SCIP_PRESOL* SCIPfindPresol(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of presolver */
   );

/** returns the array of currently available presolvers */
EXTERN
SCIP_PRESOL** SCIPgetPresols(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available presolvers */
EXTERN
int SCIPgetNPresols(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the priority of a presolver */
EXTERN
SCIP_RETCODE SCIPsetPresolPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOL*          presol,             /**< presolver */
   int                   priority            /**< new priority of the presolver */
   );

/** creates a relaxation handler and includes it in SCIP
 *
 *  @note method has all relaxation handler callbacks as arguments and is thus changed every time a new
 *        callback is added
 *        in future releases; consider using SCIPincludeRelaxBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludeRelax(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of relaxation handler */
   const char*           desc,               /**< description of relaxation handler */
   int                   priority,           /**< priority of the relaxation handler (negative: after LP, non-negative: before LP) */
   int                   freq,               /**< frequency for calling relaxation handler */
   SCIP_DECL_RELAXCOPY   ((*relaxcopy)),     /**< copy method of relaxation handler or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_RELAXFREE   ((*relaxfree)),     /**< destructor of relaxation handler */
   SCIP_DECL_RELAXINIT   ((*relaxinit)),     /**< initialize relaxation handler */
   SCIP_DECL_RELAXEXIT   ((*relaxexit)),     /**< deinitialize relaxation handler */
   SCIP_DECL_RELAXINITSOL((*relaxinitsol)),  /**< solving process initialization method of relaxation handler */
   SCIP_DECL_RELAXEXITSOL((*relaxexitsol)),  /**< solving process deinitialization method of relaxation handler */
   SCIP_DECL_RELAXEXEC   ((*relaxexec)),     /**< execution method of relaxation handler */
   SCIP_RELAXDATA*       relaxdata           /**< relaxation handler data */
   );

/** creates a relaxation handler and includes it in SCIP. All non fundamental
 *  (or optional) callbacks as, e.g., init and exit callbacks, will be set to NULL.
 *  Optional callbacks can be set via specific setter functions, see SCIPsetRelaxInit(), SCIPsetRelaxExit(),
 *  SCIPsetRelaxCopy(), SCIPsetRelaxFree(), SCIPsetRelaxInitsol(), and SCIPsetRelaxExitsol()
 *
 *  @note if you want to set all callbacks with a single method call, consider using SCIPincludeRelax() instead
 */
EXTERN
SCIP_RETCODE SCIPincludeRelaxBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAX**          relaxptr,           /**< reference to relaxation pointer, or NULL */
   const char*           name,               /**< name of relaxation handler */
   const char*           desc,               /**< description of relaxation handler */
   int                   priority,           /**< priority of the relaxation handler (negative: after LP, non-negative: before LP) */
   int                   freq,               /**< frequency for calling relaxation handler */
   SCIP_DECL_RELAXEXEC   ((*relaxexec)),     /**< execution method of relaxation handler */
   SCIP_RELAXDATA*       relaxdata           /**< relaxation handler data */
   );

/** sets copy method of relaxation handler */
EXTERN
SCIP_RETCODE SCIPsetRelaxCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAX*           relax,              /**< relaxation handler */
   SCIP_DECL_RELAXCOPY   ((*relaxcopy))      /**< copy method of relaxation handler or NULL if you don't want to copy your plugin into sub-SCIPs */
   );

/** sets destructor method of relaxation handler */
EXTERN
SCIP_RETCODE SCIPsetRelaxFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAX*           relax,              /**< relaxation handler */
   SCIP_DECL_RELAXFREE   ((*relaxfree))      /**< destructor of relaxation handler */
   );

/** sets initialization method of relaxation handler */
EXTERN
SCIP_RETCODE SCIPsetRelaxInit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAX*           relax,              /**< relaxation handler */
   SCIP_DECL_RELAXINIT   ((*relaxinit))      /**< initialize relaxation handler */
   );

/** sets deinitialization method of relaxation handler */
EXTERN
SCIP_RETCODE SCIPsetRelaxExit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAX*           relax,              /**< relaxation handler */
   SCIP_DECL_RELAXEXIT   ((*relaxexit))      /**< deinitialize relaxation handler */
   );

/** sets solving process initialization method of relaxation handler */
EXTERN
SCIP_RETCODE SCIPsetRelaxInitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAX*           relax,              /**< relaxation handler */
   SCIP_DECL_RELAXINITSOL((*relaxinitsol))   /**< solving process initialization method of relaxation handler */
   );

/** sets solving process deinitialization method of relaxation handler */
EXTERN
SCIP_RETCODE SCIPsetRelaxExitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAX*           relax,              /**< relaxation handler */
   SCIP_DECL_RELAXEXITSOL((*relaxexitsol))   /**< solving process deinitialization method of relaxation handler */
   );

/** returns the relaxation handler of the given name, or NULL if not existing */
EXTERN
SCIP_RELAX* SCIPfindRelax(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of relaxation handler*/
   );

/** returns the array of currently available relaxation handlers  */
EXTERN
SCIP_RELAX** SCIPgetRelaxs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available relaxation handlers  */
EXTERN
int SCIPgetNRelaxs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the priority of a relaxation handler*/
EXTERN
SCIP_RETCODE SCIPsetRelaxPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAX*           relax,              /**< relaxation handler */
   int                   priority            /**< new priority of the relaxation handler */
   );

/** creates a separator and includes it in SCIP.
 *
 *  @note method has all separator callbacks as arguments and is thus changed every time a new
 *        callback is added
 *        in future releases; consider using SCIPincludeSepaBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludeSepa(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of separator */
   const char*           desc,               /**< description of separator */
   int                   priority,           /**< priority of separator (>= 0: before, < 0: after constraint handlers) */
   int                   freq,               /**< frequency for calling separator */
   SCIP_Real             maxbounddist,       /**< maximal relative distance from current node's dual bound to primal bound compared
                                              *   to best node's dual bound for applying separation */
   SCIP_Bool             usessubscip,        /**< does the separator use a secondary SCIP instance? */
   SCIP_Bool             delay,              /**< should separator be delayed, if other separators found cuts? */
   SCIP_DECL_SEPACOPY    ((*sepacopy)),      /**< copy method of separator or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_SEPAFREE    ((*sepafree)),      /**< destructor of separator */
   SCIP_DECL_SEPAINIT    ((*sepainit)),      /**< initialize separator */
   SCIP_DECL_SEPAEXIT    ((*sepaexit)),      /**< deinitialize separator */
   SCIP_DECL_SEPAINITSOL ((*sepainitsol)),   /**< solving process initialization method of separator */
   SCIP_DECL_SEPAEXITSOL ((*sepaexitsol)),   /**< solving process deinitialization method of separator */
   SCIP_DECL_SEPAEXECLP  ((*sepaexeclp)),    /**< LP solution separation method of separator */
   SCIP_DECL_SEPAEXECSOL ((*sepaexecsol)),   /**< arbitrary primal solution separation method of separator */
   SCIP_SEPADATA*        sepadata            /**< separator data */
   );

/** creates a separator and includes it in SCIP with its most fundamental callbacks. All non-fundamental
 *  (or optional) callbacks as, e.g., init and exit callbacks, will be set to NULL.
 *  Optional callbacks can be set via specific setter functions, see SCIPsetSepaInit(), SCIPsetSepaFree(),
 *  SCIPsetSepaInitsol(), SCIPsetSepaExitsol(), SCIPsetSepaCopy(), SCIPsetExit().
 *
 *  @note if you want to set all callbacks with a single method call, consider using SCIPincludeSepa() instead
 */
EXTERN
SCIP_RETCODE SCIPincludeSepaBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA**           sepa,               /**< reference to a separator, or NULL */
   const char*           name,               /**< name of separator */
   const char*           desc,               /**< description of separator */
   int                   priority,           /**< priority of separator (>= 0: before, < 0: after constraint handlers) */
   int                   freq,               /**< frequency for calling separator */
   SCIP_Real             maxbounddist,       /**< maximal relative distance from current node's dual bound to primal bound compared
                                              *   to best node's dual bound for applying separation */
   SCIP_Bool             usessubscip,        /**< does the separator use a secondary SCIP instance? */
   SCIP_Bool             delay,              /**< should separator be delayed, if other separators found cuts? */
   SCIP_DECL_SEPAEXECLP  ((*sepaexeclp)),    /**< LP solution separation method of separator */
   SCIP_DECL_SEPAEXECSOL ((*sepaexecsol)),   /**< arbitrary primal solution separation method of separator */
   SCIP_SEPADATA*        sepadata            /**< separator data */
   );

/** sets copy method of separator */
EXTERN
SCIP_RETCODE SCIPsetSepaCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_DECL_SEPACOPY    ((*sepacopy))       /**< copy method of separator or NULL if you don't want to copy your plugin into sub-SCIPs */
   );

/** sets destructor method of separator */
EXTERN
SCIP_RETCODE SCIPsetSepaFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_DECL_SEPAFREE    ((*sepafree))       /**< destructor of separator */
   );

/** sets initialization method of separator */
EXTERN
SCIP_RETCODE SCIPsetSepaInit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_DECL_SEPAINIT    ((*sepainit))       /**< initialize separator */
   );

/** sets deinitialization method of separator */
EXTERN
SCIP_RETCODE SCIPsetSepaExit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_DECL_SEPAEXIT    ((*sepaexit))       /**< deinitialize separator */
   );

/** sets solving process initialization method of separator */
EXTERN
SCIP_RETCODE SCIPsetSepaInitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_DECL_SEPAINITSOL ((*sepainitsol))    /**< solving process initialization method of separator */
   );

/** sets solving process deinitialization method of separator */
EXTERN
SCIP_RETCODE SCIPsetSepaExitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_DECL_SEPAEXITSOL ((*sepaexitsol))    /**< solving process deinitialization method of separator */
   );

/** returns the separator of the given name, or NULL if not existing */
EXTERN
SCIP_SEPA* SCIPfindSepa(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of separator */
   );

/** returns the array of currently available separators */
EXTERN
SCIP_SEPA** SCIPgetSepas(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available separators */
EXTERN
int SCIPgetNSepas(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the priority of a separator */
EXTERN
SCIP_RETCODE SCIPsetSepaPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   int                   priority            /**< new priority of the separator */
   );

/** creates a propagator and includes it in SCIP.
 *
 *  @note method has all propagator callbacks as arguments and is thus changed every time a new
 *        callback is added
 *        in future releases; consider using SCIPincludePropBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludeProp(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of propagator */
   const char*           desc,               /**< description of propagator */
   int                   priority,           /**< priority of the propagator (>= 0: before, < 0: after constraint handlers) */
   int                   freq,               /**< frequency for calling propagator */
   SCIP_Bool             delay,              /**< should propagator be delayed, if other propagators found reductions? */
   SCIP_PROPTIMING       timingmask,         /**< positions in the node solving loop where propagators should be executed */
   int                   presolpriority,     /**< priority of the propagator (>= 0: before, < 0: after constraint handlers) */
   int                   presolmaxrounds,    /**< maximal number of presolving rounds the propagator participates in (-1: no limit) */
   SCIP_Bool             presoldelay,        /**< should presolving be delayed, if other presolvers found reductions? */
   SCIP_DECL_PROPCOPY    ((*propcopy)),      /**< copy method of propagator or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_PROPFREE    ((*propfree)),      /**< destructor of propagator */
   SCIP_DECL_PROPINIT    ((*propinit)),      /**< initialize propagator */
   SCIP_DECL_PROPEXIT    ((*propexit)),      /**< deinitialize propagator */
   SCIP_DECL_PROPINITPRE ((*propinitpre)),   /**< presolving initialization method of propagator */
   SCIP_DECL_PROPEXITPRE ((*propexitpre)),   /**< presolving deinitialization method of propagator */
   SCIP_DECL_PROPINITSOL ((*propinitsol)),   /**< solving process initialization method of propagator */
   SCIP_DECL_PROPEXITSOL ((*propexitsol)),   /**< solving process deinitialization method of propagator */
   SCIP_DECL_PROPPRESOL  ((*proppresol)),    /**< presolving method */
   SCIP_DECL_PROPEXEC    ((*propexec)),      /**< execution method of propagator */
   SCIP_DECL_PROPRESPROP ((*propresprop)),   /**< propagation conflict resolving method */
   SCIP_PROPDATA*        propdata            /**< propagator data */
   );

/** creates a propagator and includes it in SCIP. All non-fundamental (or optional) callbacks will be set to NULL.
 *  Optional callbacks can be set via specific setter functions, see SCIPsetPropInit(), SCIPsetPropExit(),
 *  SCIPsetPropCopy(), SCIPsetPropFree(), SCIPsetPropInitsol(), SCIPsetPropExitsol(),
 *  SCIPsetPropInitpre(), SCIPsetPropExitpre(), SCIPsetPropPresol(), and SCIPsetPropResprop().
 *
 *  @note if you want to set all callbacks with a single method call, consider using SCIPincludeProp() instead
 */
EXTERN
SCIP_RETCODE SCIPincludePropBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP**           propptr,            /**< reference to a propagator pointer, or NULL */
   const char*           name,               /**< name of propagator */
   const char*           desc,               /**< description of propagator */
   int                   priority,           /**< priority of the propagator (>= 0: before, < 0: after constraint handlers) */
   int                   freq,               /**< frequency for calling propagator */
   SCIP_Bool             delay,              /**< should propagator be delayed, if other propagators found reductions? */
   SCIP_PROPTIMING       timingmask,         /**< positions in the node solving loop where propagators should be executed */
   SCIP_DECL_PROPEXEC    ((*propexec)),      /**< execution method of propagator */
   SCIP_PROPDATA*        propdata            /**< propagator data */
   );

/** sets copy method of propagator */
EXTERN
SCIP_RETCODE SCIPsetPropCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   SCIP_DECL_PROPCOPY    ((*propcopy))       /**< copy method of propagator or NULL if you don't want to copy your plugin into sub-SCIPs */
   );

/** sets destructor method of propagator */
EXTERN
SCIP_RETCODE SCIPsetPropFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   SCIP_DECL_PROPFREE    ((*propfree))       /**< destructor of propagator */
   );

/** sets initialization method of propagator */
EXTERN
SCIP_RETCODE SCIPsetPropInit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   SCIP_DECL_PROPINIT    ((*propinit))       /**< initialize propagator */
   );

/** sets deinitialization method of propagator */
EXTERN
SCIP_RETCODE SCIPsetPropExit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   SCIP_DECL_PROPEXIT    ((*propexit))       /**< deinitialize propagator */
   );

/** sets solving process initialization method of propagator */
EXTERN
SCIP_RETCODE SCIPsetPropInitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   SCIP_DECL_PROPINITSOL((*propinitsol))     /**< solving process initialization method of propagator */
   );

/** sets solving process deinitialization method of propagator */
EXTERN
SCIP_RETCODE SCIPsetPropExitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   SCIP_DECL_PROPEXITSOL ((*propexitsol))    /**< solving process deinitialization method of propagator */
   );

/** sets preprocessing initialization method of propagator */
EXTERN
SCIP_RETCODE SCIPsetPropInitpre(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   SCIP_DECL_PROPINITPRE((*propinitpre))     /**< preprocessing initialization method of propagator */
   );

/** sets preprocessing deinitialization method of propagator */
EXTERN
SCIP_RETCODE SCIPsetPropExitpre(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   SCIP_DECL_PROPEXITPRE((*propexitpre))     /**< preprocessing deinitialization method of propagator */
   );

/** sets presolving method of propagator */
EXTERN
SCIP_RETCODE SCIPsetPropPresol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   SCIP_DECL_PROPPRESOL((*proppresol)),      /**< presolving method of propagator */
   int                   presolpriority,     /**< presolving priority of the propagator (>= 0: before, < 0: after constraint handlers) */
   int                   presolmaxrounds,    /**< maximal number of presolving rounds the propagator participates in (-1: no limit) */
   SCIP_Bool             presoldelay         /**< should presolving be delayed, if other presolvers found reductions? */
   );

/** sets propagation conflict resolving callback of propagator */
EXTERN
SCIP_RETCODE SCIPsetPropResprop(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   SCIP_DECL_PROPRESPROP ((*propresprop))    /**< propagation conflict resolving callback */
   );

/** returns the propagator of the given name, or NULL if not existing */
EXTERN
SCIP_PROP* SCIPfindProp(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of propagator */
   );

/** returns the array of currently available propagators */
EXTERN
SCIP_PROP** SCIPgetProps(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available propagators */
EXTERN
int SCIPgetNProps(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the priority of a propagator */
EXTERN
SCIP_RETCODE SCIPsetPropPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   int                   priority            /**< new priority of the propagator */
   );

/** sets the presolving priority of a propagator */
EXTERN
SCIP_RETCODE SCIPsetPropPresolPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< propagator */
   int                   presolpriority      /**< new presol priority of the propagator */
   );


/** creates a primal heuristic and includes it in SCIP.
 *
 *  @note method has all heuristic callbacks as arguments and is thus changed every time a new
 *        callback is added in future releases; consider using SCIPincludeHeurBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludeHeur(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of primal heuristic */
   const char*           desc,               /**< description of primal heuristic */
   char                  dispchar,           /**< display character of primal heuristic */
   int                   priority,           /**< priority of the primal heuristic */
   int                   freq,               /**< frequency for calling primal heuristic */
   int                   freqofs,            /**< frequency offset for calling primal heuristic */
   int                   maxdepth,           /**< maximal depth level to call heuristic at (-1: no limit) */
   unsigned int          timingmask,         /**< positions in the node solving loop where heuristic should be executed;
                                              *   see definition of SCIP_HeurTiming for possible values */
   SCIP_Bool             usessubscip,        /**< does the heuristic use a secondary SCIP instance? */
   SCIP_DECL_HEURCOPY    ((*heurcopy)),      /**< copy method of primal heuristic or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_HEURFREE    ((*heurfree)),      /**< destructor of primal heuristic */
   SCIP_DECL_HEURINIT    ((*heurinit)),      /**< initialize primal heuristic */
   SCIP_DECL_HEUREXIT    ((*heurexit)),      /**< deinitialize primal heuristic */
   SCIP_DECL_HEURINITSOL ((*heurinitsol)),   /**< solving process initialization method of primal heuristic */
   SCIP_DECL_HEUREXITSOL ((*heurexitsol)),   /**< solving process deinitialization method of primal heuristic */
   SCIP_DECL_HEUREXEC    ((*heurexec)),      /**< execution method of primal heuristic */
   SCIP_HEURDATA*        heurdata            /**< primal heuristic data */
   );

/** creates a primal heuristic and includes it in SCIP with its most fundamental callbacks.
 *  All non-fundamental (or optional) callbacks
 *  as, e. g., init and exit callbacks, will be set to NULL.
 *  Optional callbacks can be set via specific setter functions, see SCIPsetHeurCopy(), SCIPsetHeurFree(),
 *  SCIPsetHeurInit(), SCIPsetHeurExit(), SCIPsetHeurInitsol(), and SCIPsetHeurExitsol()
 *
*  @note if you want to set all callbacks with a single method call, consider using SCIPincludeHeur() instead
 */
EXTERN
SCIP_RETCODE SCIPincludeHeurBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR**           heur,               /**< pointer to the heuristic */
   const char*           name,               /**< name of primal heuristic */
   const char*           desc,               /**< description of primal heuristic */
   char                  dispchar,           /**< display character of primal heuristic */
   int                   priority,           /**< priority of the primal heuristic */
   int                   freq,               /**< frequency for calling primal heuristic */
   int                   freqofs,            /**< frequency offset for calling primal heuristic */
   int                   maxdepth,           /**< maximal depth level to call heuristic at (-1: no limit) */
   unsigned int          timingmask,         /**< positions in the node solving loop where heuristic should be executed;
                                              *   see definition of SCIP_HeurTiming for possible values */
   SCIP_Bool             usessubscip,        /**< does the heuristic use a secondary SCIP instance? */
   SCIP_DECL_HEUREXEC    ((*heurexec)),      /**< execution method of primal heuristic */
   SCIP_HEURDATA*        heurdata            /**< primal heuristic data */
   );

/** sets copy method of primal heuristic */
EXTERN
SCIP_RETCODE SCIPsetHeurCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_DECL_HEURCOPY    ((*heurcopy))       /**< copy method of primal heuristic or NULL if you don't want to copy your plugin into sub-SCIPs */
   );

/** sets destructor method of primal heuristic */
EXTERN
SCIP_RETCODE SCIPsetHeurFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_DECL_HEURFREE    ((*heurfree))       /**< destructor of primal heuristic */
   );

/** sets initialization method of primal heuristic */
EXTERN
SCIP_RETCODE SCIPsetHeurInit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_DECL_HEURINIT    ((*heurinit))       /**< initialize primal heuristic */
   );

/** sets deinitialization method of primal heuristic */
EXTERN
SCIP_RETCODE SCIPsetHeurExit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_DECL_HEUREXIT    ((*heurexit))       /**< deinitialize primal heuristic */
   );

/** sets solving process initialization method of primal heuristic */
EXTERN
SCIP_RETCODE SCIPsetHeurInitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_DECL_HEURINITSOL ((*heurinitsol))    /**< solving process initialization method of primal heuristic */
   );

/** sets solving process deinitialization method of primal heuristic */
EXTERN
SCIP_RETCODE SCIPsetHeurExitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_DECL_HEUREXITSOL ((*heurexitsol))    /**< solving process deinitialization method of primal heuristic */
   );

/** returns the primal heuristic of the given name, or NULL if not existing */
EXTERN
SCIP_HEUR* SCIPfindHeur(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of primal heuristic */
   );

/** returns the array of currently available primal heuristics */
EXTERN
SCIP_HEUR** SCIPgetHeurs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available primal heuristics */
EXTERN
int SCIPgetNHeurs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the priority of a primal heuristic */
EXTERN
SCIP_RETCODE SCIPsetHeurPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< primal heuristic */
   int                   priority            /**< new priority of the primal heuristic */
   );

/** creates an event handler and includes it in SCIP
 *
 *  @note method has all event handler callbacks as arguments and is thus changed every time a new
 *        callback is added in future releases; consider using SCIPincludeEventhdlrBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludeEventhdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of event handler */
   const char*           desc,               /**< description of event handler */
   SCIP_DECL_EVENTCOPY   ((*eventcopy)),     /**< copy method of event handler or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_EVENTFREE   ((*eventfree)),     /**< destructor of event handler */
   SCIP_DECL_EVENTINIT   ((*eventinit)),     /**< initialize event handler */
   SCIP_DECL_EVENTEXIT   ((*eventexit)),     /**< deinitialize event handler */
   SCIP_DECL_EVENTINITSOL((*eventinitsol)),  /**< solving process initialization method of event handler */
   SCIP_DECL_EVENTEXITSOL((*eventexitsol)),  /**< solving process deinitialization method of event handler */
   SCIP_DECL_EVENTDELETE ((*eventdelete)),   /**< free specific event data */
   SCIP_DECL_EVENTEXEC   ((*eventexec)),     /**< execute event handler */
   SCIP_EVENTHDLRDATA*   eventhdlrdata       /**< event handler data */
   );

/** creates an event handler and includes it in SCIP with all its non-fundamental callbacks set
 *  to NULL; if needed, non-fundamental callbacks can be set afterwards via setter functions
 *  SCIPsetEventhdlrCopy(), SCIPsetEventhdlrFree(), SCIPsetEventhdlrInit(), SCIPsetEventhdlrExit(),
 *  SCIPsetEventhdlrInitsol(), SCIPsetEventhdlrExitsol(), and SCIPsetEventhdlrDelete()
 *
 *  @note if you want to set all callbacks with a single method call, consider using SCIPincludeEventhdlr() instead
 */
EXTERN
SCIP_RETCODE SCIPincludeEventhdlrBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EVENTHDLR**      eventhdlrptr,       /**< reference to an event handler, or NULL */
   const char*           name,               /**< name of event handler */
   const char*           desc,               /**< description of event handler */
   SCIP_DECL_EVENTEXEC   ((*eventexec)),     /**< execute event handler */
   SCIP_EVENTHDLRDATA*   eventhdlrdata       /**< event handler data */
   );

/** sets copy callback of the event handler */
EXTERN
SCIP_RETCODE SCIPsetEventhdlrCopy(
   SCIP*                 scip,               /**< scip instance */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler */
   SCIP_DECL_EVENTCOPY   ((*eventcopy))      /**< copy callback of the event handler */
   );

/** sets deinitialization callback of the event handler */
EXTERN
SCIP_RETCODE SCIPsetEventhdlrFree(
   SCIP*                 scip,               /**< scip instance */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler */
   SCIP_DECL_EVENTFREE   ((*eventfree))      /**< deinitialization callback of the event handler */
   );

/** sets initialization callback of the event handler */
EXTERN
SCIP_RETCODE SCIPsetEventhdlrInit(
   SCIP*                 scip,               /**< scip instance */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler */
   SCIP_DECL_EVENTINIT   ((*eventinit))      /**< initialize event handler */
   );

/** sets deinitialization callback of the event handler */
EXTERN
SCIP_RETCODE SCIPsetEventhdlrExit(
   SCIP*                 scip,               /**< scip instance */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler */
   SCIP_DECL_EVENTEXIT   ((*eventexit))      /**< deinitialize event handler */
   );

/** sets solving process initialization callback of the event handler */
EXTERN
SCIP_RETCODE SCIPsetEventhdlrInitsol(
   SCIP*                 scip,               /**< scip instance */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler */
   SCIP_DECL_EVENTINITSOL((*eventinitsol))   /**< solving process initialization callback of event handler */
   );

/** sets solving process deinitialization callback of the event handler */
EXTERN
SCIP_RETCODE SCIPsetEventhdlrExitsol(
   SCIP*                 scip,               /**< scip instance */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler */
   SCIP_DECL_EVENTEXITSOL((*eventexitsol))   /**< solving process deinitialization callback of event handler */
   );

/** sets callback of the event handler to free specific event data */
EXTERN
SCIP_RETCODE SCIPsetEventhdlrDelete(
   SCIP*                 scip,               /**< scip instance */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler */
   SCIP_DECL_EVENTDELETE ((*eventdelete))    /**< free specific event data */
   );

/** returns the event handler of the given name, or NULL if not existing */
EXTERN
SCIP_EVENTHDLR* SCIPfindEventhdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of event handler */
   );

/** returns the array of currently available event handlers */
EXTERN
SCIP_EVENTHDLR** SCIPgetEventhdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available event handlers */
EXTERN
int SCIPgetNEventhdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** creates a node selector and includes it in SCIP.
 *
 *  @note method has all node selector callbacks as arguments and is thus changed every time a new
 *        callback is added in future releases; consider using SCIPincludeNodeselBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludeNodesel(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of node selector */
   const char*           desc,               /**< description of node selector */
   int                   stdpriority,        /**< priority of the node selector in standard mode */
   int                   memsavepriority,    /**< priority of the node selector in memory saving mode */
   SCIP_DECL_NODESELCOPY ((*nodeselcopy)),   /**< copy method of node selector or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_NODESELFREE ((*nodeselfree)),   /**< destructor of node selector */
   SCIP_DECL_NODESELINIT ((*nodeselinit)),   /**< initialize node selector */
   SCIP_DECL_NODESELEXIT ((*nodeselexit)),   /**< deinitialize node selector */
   SCIP_DECL_NODESELINITSOL((*nodeselinitsol)),/**< solving process initialization method of node selector */
   SCIP_DECL_NODESELEXITSOL((*nodeselexitsol)),/**< solving process deinitialization method of node selector */
   SCIP_DECL_NODESELSELECT((*nodeselselect)),/**< node selection method */
   SCIP_DECL_NODESELCOMP ((*nodeselcomp)),   /**< node comparison method */
   SCIP_NODESELDATA*     nodeseldata         /**< node selector data */
   );

/** Creates a node selector and includes it in SCIP with its most fundamental callbacks. All non-fundamental
 *  (or optional) callbacks as, e.g., init and exit callbacks, will be set to NULL.
 *  Optional callbacks can be set via specific setter functions, see SCIPsetNodeselCopy(), SCIPsetNodeselFree(),
 *  SCIPsetNodeselInit(), SCIPsetNodeselExit(), SCIPsetNodeselInitsol(), and SCIPsetNodeselExitsol()
 *
 *  @note if you want to set all callbacks with a single method call, consider using SCIPincludeNodesel() instead
 */
EXTERN
SCIP_RETCODE SCIPincludeNodeselBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODESEL**        nodesel,            /**< reference to a node selector, or NULL */
   const char*           name,               /**< name of node selector */
   const char*           desc,               /**< description of node selector */
   int                   stdpriority,        /**< priority of the node selector in standard mode */
   int                   memsavepriority,    /**< priority of the node selector in memory saving mode */
   SCIP_DECL_NODESELSELECT((*nodeselselect)),/**< node selection method */
   SCIP_DECL_NODESELCOMP ((*nodeselcomp)),   /**< node comparison method */
   SCIP_NODESELDATA*     nodeseldata         /**< node selector data */
   );

/** sets copy method of node selector */
EXTERN
SCIP_RETCODE SCIPsetNodeselCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODESEL*         nodesel,            /**< node selector */
   SCIP_DECL_NODESELCOPY ((*nodeselcopy))    /**< copy method of node selector or NULL if you don't want to copy your plugin into sub-SCIPs */
   );

/** sets destructor method of node selector */
EXTERN
SCIP_RETCODE SCIPsetNodeselFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODESEL*         nodesel,            /**< node selector */
   SCIP_DECL_NODESELFREE ((*nodeselfree))    /**< destructor of node selector */
   );

/** sets initialization method of node selector */
EXTERN
SCIP_RETCODE SCIPsetNodeselInit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODESEL*         nodesel,            /**< node selector */
   SCIP_DECL_NODESELINIT ((*nodeselinit))    /**< initialize node selector */
   );

/** sets deinitialization method of node selector */
EXTERN
SCIP_RETCODE SCIPsetNodeselExit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODESEL*         nodesel,            /**< node selector */
   SCIP_DECL_NODESELEXIT ((*nodeselexit))    /**< deinitialize node selector */
   );

/** sets solving process initialization method of node selector */
EXTERN
SCIP_RETCODE SCIPsetNodeselInitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODESEL*         nodesel,            /**< node selector */
   SCIP_DECL_NODESELINITSOL ((*nodeselinitsol))/**< solving process initialization method of node selector */
   );

/** sets solving process deinitialization method of node selector */
EXTERN
SCIP_RETCODE SCIPsetNodeselExitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODESEL*         nodesel,            /**< node selector */
   SCIP_DECL_NODESELEXITSOL ((*nodeselexitsol))/**< solving process deinitialization method of node selector */
   );

/** returns the node selector of the given name, or NULL if not existing */
EXTERN
SCIP_NODESEL* SCIPfindNodesel(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of node selector */
   );

/** returns the array of currently available node selectors */
EXTERN
SCIP_NODESEL** SCIPgetNodesels(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available node selectors */
EXTERN
int SCIPgetNNodesels(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the priority of a node selector in standard mode */
EXTERN
SCIP_RETCODE SCIPsetNodeselStdPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODESEL*         nodesel,            /**< node selector */
   int                   priority            /**< new standard priority of the node selector */
   );

/** sets the priority of a node selector in memory saving mode */
EXTERN
SCIP_RETCODE SCIPsetNodeselMemsavePriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODESEL*         nodesel,            /**< node selector */
   int                   priority            /**< new memory saving priority of the node selector */
   );

/** returns the currently used node selector */
EXTERN
SCIP_NODESEL* SCIPgetNodesel(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** creates a branching rule and includes it in SCIP
 *
 *  @note method has all branching rule callbacks as arguments and is thus changed every time a new
 *        callback is added in future releases; consider using SCIPincludeBranchruleBasic() and setter functions
 *        if you seek for a method which is less likely to change in future releases
 */
EXTERN
SCIP_RETCODE SCIPincludeBranchrule(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of branching rule */
   const char*           desc,               /**< description of branching rule */
   int                   priority,           /**< priority of the branching rule */
   int                   maxdepth,           /**< maximal depth level, up to which this branching rule should be used (or -1) */
   SCIP_Real             maxbounddist,       /**< maximal relative distance from current node's dual bound to primal bound
                                              *   compared to best node's dual bound for applying branching rule
                                              *   (0.0: only on current best node, 1.0: on all nodes) */
   SCIP_DECL_BRANCHCOPY  ((*branchcopy)),    /**< copy method of branching rule or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_BRANCHFREE  ((*branchfree)),    /**< destructor of branching rule */
   SCIP_DECL_BRANCHINIT  ((*branchinit)),    /**< initialize branching rule */
   SCIP_DECL_BRANCHEXIT  ((*branchexit)),    /**< deinitialize branching rule */
   SCIP_DECL_BRANCHINITSOL((*branchinitsol)),/**< solving process initialization method of branching rule */
   SCIP_DECL_BRANCHEXITSOL((*branchexitsol)),/**< solving process deinitialization method of branching rule */
   SCIP_DECL_BRANCHEXECLP((*branchexeclp)),  /**< branching execution method for fractional LP solutions */
   SCIP_DECL_BRANCHEXECEXT((*branchexecext)),/**< branching execution method for external candidates */
   SCIP_DECL_BRANCHEXECPS((*branchexecps)),  /**< branching execution method for not completely fixed pseudo solutions */
   SCIP_BRANCHRULEDATA*  branchruledata      /**< branching rule data */
   );

/** creates a branching rule and includes it in SCIP. All non-fundamental (or optional) callbacks will be set to NULL.
 *  Optional callbacks can be set via specific setter functions, see SCIPsetBranchruleInit(), SCIPsetBranchruleExit(),
 *  SCIPsetBranchruleCopy(), SCIPsetBranchruleFree(), SCIPsetBranchruleInitsol(), SCIPsetBranchruleExitsol(),
 *  SCIPsetBranchruleExecLp(), SCIPsetBranchruleExecExt(), and SCIPsetBranchruleExecPs().
 *
 *  @note if you want to set all callbacks with a single method call, consider using SCIPincludeBranchrule() instead
 */
EXTERN
SCIP_RETCODE SCIPincludeBranchruleBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE**     branchruleptr,      /**< pointer to branching rule, or NULL */
   const char*           name,               /**< name of branching rule */
   const char*           desc,               /**< description of branching rule */
   int                   priority,           /**< priority of the branching rule */
   int                   maxdepth,           /**< maximal depth level, up to which this branching rule should be used (or -1) */
   SCIP_Real             maxbounddist,       /**< maximal relative distance from current node's dual bound to primal bound
                                              *   compared to best node's dual bound for applying branching rule
                                              *   (0.0: only on current best node, 1.0: on all nodes) */
   SCIP_BRANCHRULEDATA*  branchruledata      /**< branching rule data */
   );

/** sets copy method of branching rule */
EXTERN
SCIP_RETCODE SCIPsetBranchruleCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_DECL_BRANCHCOPY  ((*branchcopy))     /**< copy method of branching rule or NULL if you don't want to copy your plugin into sub-SCIPs */
   );

/** sets destructor method of branching rule */
EXTERN
SCIP_RETCODE SCIPsetBranchruleFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_DECL_BRANCHFREE  ((*branchfree))     /**< destructor of branching rule */
   );

/** sets initialization method of branching rule */
EXTERN
SCIP_RETCODE SCIPsetBranchruleInit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_DECL_BRANCHINIT  ((*branchinit))     /**< initialize branching rule */
   );

/** sets deinitialization method of branching rule */
EXTERN
SCIP_RETCODE SCIPsetBranchruleExit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_DECL_BRANCHEXIT  ((*branchexit))     /**< deinitialize branching rule */
   );

/** sets solving process initialization method of branching rule */
EXTERN
SCIP_RETCODE SCIPsetBranchruleInitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_DECL_BRANCHINITSOL((*branchinitsol)) /**< solving process initialization method of branching rule */
   );

/** sets solving process deinitialization method of branching rule */
EXTERN
SCIP_RETCODE SCIPsetBranchruleExitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_DECL_BRANCHEXITSOL((*branchexitsol)) /**< solving process deinitialization method of branching rule */
   );

/** sets branching execution method for fractional LP solutions */
EXTERN
SCIP_RETCODE SCIPsetBranchruleExecLp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_DECL_BRANCHEXECLP((*branchexeclp))   /**< branching execution method for fractional LP solutions */
   );

/** sets branching execution method for external candidates  */
EXTERN
SCIP_RETCODE SCIPsetBranchruleExecExt(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_DECL_BRANCHEXECEXT((*branchexecext)) /**< branching execution method for external candidates */
   );

/** sets branching execution method for not completely fixed pseudo solutions */
EXTERN
SCIP_RETCODE SCIPsetBranchruleExecPs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_DECL_BRANCHEXECPS((*branchexecps))   /**< branching execution method for not completely fixed pseudo solutions */
   );

/** returns the branching rule of the given name, or NULL if not existing */
EXTERN
SCIP_BRANCHRULE* SCIPfindBranchrule(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of branching rule */
   );

/** returns the array of currently available branching rules */
EXTERN
SCIP_BRANCHRULE** SCIPgetBranchrules(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available branching rules */
EXTERN
int SCIPgetNBranchrules(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the priority of a branching rule */
EXTERN
SCIP_RETCODE SCIPsetBranchrulePriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   int                   priority            /**< new priority of the branching rule */
   );

/** sets maximal depth level, up to which this branching rule should be used (-1 for no limit) */
EXTERN
SCIP_RETCODE SCIPsetBranchruleMaxdepth(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   int                   maxdepth            /**< new maxdepth of the branching rule */
   );

/** sets maximal relative distance from current node's dual bound to primal bound for applying branching rule */
EXTERN
SCIP_RETCODE SCIPsetBranchruleMaxbounddist(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_Real             maxbounddist        /**< new maxbounddist of the branching rule */
   );

/** creates a display column and includes it in SCIP */
EXTERN
SCIP_RETCODE SCIPincludeDisp(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of display column */
   const char*           desc,               /**< description of display column */
   const char*           header,             /**< head line of display column */
   SCIP_DISPSTATUS       dispstatus,         /**< display activation status of display column */
   SCIP_DECL_DISPCOPY    ((*dispcopy)),      /**< copy method of display column or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_DISPFREE    ((*dispfree)),      /**< destructor of display column */
   SCIP_DECL_DISPINIT    ((*dispinit)),      /**< initialize display column */
   SCIP_DECL_DISPEXIT    ((*dispexit)),      /**< deinitialize display column */
   SCIP_DECL_DISPINITSOL ((*dispinitsol)),   /**< solving process initialization method of display column */
   SCIP_DECL_DISPEXITSOL ((*dispexitsol)),   /**< solving process deinitialization method of display column */
   SCIP_DECL_DISPOUTPUT  ((*dispoutput)),    /**< output method */
   SCIP_DISPDATA*        dispdata,           /**< display column data */
   int                   width,              /**< width of display column (no. of chars used) */
   int                   priority,           /**< priority of display column */
   int                   position,           /**< relative position of display column */
   SCIP_Bool             stripline           /**< should the column be separated with a line from its right neighbor? */
   );

/** returns the display column of the given name, or NULL if not existing */
EXTERN
SCIP_DISP* SCIPfindDisp(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of display column */
   );

/** returns the array of currently available display columns */
EXTERN
SCIP_DISP** SCIPgetDisps(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available display columns */
EXTERN
int SCIPgetNDisps(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** automatically selects display columns for being shown w.r.t. the display width parameter */
EXTERN
SCIP_RETCODE SCIPautoselectDisps(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** includes an NLPI in SCIP */
EXTERN
SCIP_RETCODE SCIPincludeNlpi(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPI*            nlpi                /**< NLPI data structure */
   );

/** returns the NLPI of the given name, or NULL if not existing */
EXTERN
SCIP_NLPI* SCIPfindNlpi(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of NLPI */
   );

/** returns the array of currently available NLPIs (sorted by priority) */
EXTERN
SCIP_NLPI** SCIPgetNlpis(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently available NLPIs */
EXTERN
int SCIPgetNNlpis(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the priority of an NLPI */
EXTERN
SCIP_RETCODE SCIPsetNlpiPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPI*            nlpi,               /**< NLPI */
   int                   priority            /**< new priority of the NLPI */
   );

/** includes information about an external code linked into the SCIP library */
EXTERN
SCIP_RETCODE SCIPincludeExternalCodeInformation(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of external code */
   const char*           description         /**< description of external code, or NULL */
   );

/** returns an array of names of currently included external codes */
EXTERN
char** SCIPgetExternalCodeNames(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns an array of the descriptions of currently included external codes
 *
 *  @note some descriptions may be NULL
 */
EXTERN
char** SCIPgetExternalCodeDescriptions(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of currently included information on external codes */
EXTERN
int SCIPgetNExternalCodes(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** prints information on external codes to a file stream via the message handler system
 *
 *  @note If the message handler is set to a NULL pointer nothing will be printed
 */
EXTERN
void SCIPprintExternalCodes(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/**@} */




/*
 * user interactive dialog methods
 */

/**@name User Interactive Dialog Methods */
/**@{ */

/** creates and includes dialog
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPincludeDialog(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG**         dialog,             /**< pointer to store the dialog */
   SCIP_DECL_DIALOGCOPY  ((*dialogcopy)),    /**< copy method of dialog or NULL if you don't want to copy your plugin into sub-SCIPs */
   SCIP_DECL_DIALOGEXEC  ((*dialogexec)),    /**< execution method of dialog */
   SCIP_DECL_DIALOGDESC  ((*dialogdesc)),    /**< description output method of dialog, or NULL */
   SCIP_DECL_DIALOGFREE  ((*dialogfree)),    /**< destructor of dialog to free user data, or NULL */
   const char*           name,               /**< name of dialog: command name appearing in parent's dialog menu */
   const char*           desc,               /**< description of dialog used if description output method is NULL */
   SCIP_Bool             issubmenu,          /**< is the dialog a submenu? */
   SCIP_DIALOGDATA*      dialogdata          /**< user defined dialog data */
   );

/** returns if the dialog already exists
 *
 *  @return TRUE is returned if the dialog exits, otherwise FALSE.
 */
EXTERN
SCIP_Bool SCIPexistsDialog(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG*          dialog              /**< dialog */
   );

/** captures a dialog
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPcaptureDialog(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG*          dialog              /**< dialog */
   );

/** releases a dialog
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPreleaseDialog(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG**         dialog              /**< pointer to the dialog */
   );

/** makes given dialog the root dialog of SCIP's interactive user shell; captures dialog and releases former root dialog
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetRootDialog(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG*          dialog              /**< dialog to be the root */
   );

/** returns the root dialog of SCIP's interactive user shell
 *
 *  @return the root dialog of SCIP's interactive user shell is returned.
 */
EXTERN
SCIP_DIALOG* SCIPgetRootDialog(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** adds a sub dialog to the given dialog as menu entry and captures it
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPaddDialogEntry(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG*          dialog,             /**< dialog to extend, or NULL for root dialog */
   SCIP_DIALOG*          subdialog           /**< subdialog to add as menu entry in dialog */
   );

/** adds a single line of input which is treated as if the user entered the command line
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPaddDialogInputLine(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           inputline           /**< input line to add */
   );

/** adds a single line of input to the command history which can be accessed with the cursor keys
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPaddDialogHistoryLine(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           inputline           /**< input line to add */
   );

/** starts interactive mode of SCIP by executing the root dialog
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_FREE
 *
 *  @post After calling this method \SCIP reaches one of the following stages depending on if and when the
 *        interactive shell was closed:
 *        - \ref SCIP_STAGE_PROBLEM if the interactive shell was closed after the problem was created
 *        - \ref SCIP_STAGE_TRANSFORMED if the interactive shell was closed after the problem was transformed
 *        - \ref SCIP_STAGE_PRESOLVING if the interactive shell was closed  during presolving
 *        - \ref SCIP_STAGE_PRESOLVED if the interactive shell was closed after presolve
 *        - \ref SCIP_STAGE_SOLVING if the interactive shell was closed during the tree search
 *        - \ref SCIP_STAGE_SOLVED if the interactive shell was closed after the problem was solved
 *        - \ref SCIP_STAGE_FREE if the interactive shell was closed after the problem was freed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPstartInteraction(
   SCIP*                 scip                /**< SCIP data structure */
   );
   
/**@} */




/*
 * global problem methods
 */

/**@name Global Problem Methods */
/**@{ */

/** creates empty problem and initializes all solving data structures (the objective sense is set to MINIMIZE)
 *  If the problem type requires the use of variable pricers, these pricers should be added to the problem with calls
 *  to SCIPactivatePricer(). These pricers are automatically deactivated, when the problem is freed.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_FREE
 *
 *  @post After calling this method, \SCIP reaches the following stage:
 *        - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPcreateProb(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< problem name */
   SCIP_DECL_PROBDELORIG ((*probdelorig)),   /**< frees user data of original problem */
   SCIP_DECL_PROBTRANS   ((*probtrans)),     /**< creates user data of transformed problem by transforming original user data */
   SCIP_DECL_PROBDELTRANS((*probdeltrans)),  /**< frees user data of transformed problem */
   SCIP_DECL_PROBINITSOL ((*probinitsol)),   /**< solving process initialization method of transformed data */
   SCIP_DECL_PROBEXITSOL ((*probexitsol)),   /**< solving process deinitialization method of transformed data */
   SCIP_DECL_PROBCOPY    ((*probcopy)),      /**< copies user data if you want to copy it to a subscip, or NULL */
   SCIP_PROBDATA*        probdata            /**< user problem data set by the reader */
   );

/** creates empty problem and initializes all solving data structures (the objective sense is set to MINIMIZE)
 *  all callback methods will be set to NULL and can be set afterwards, if needed, via SCIPsetProbDelorig(),
 *  SCIPsetProbTrans(), SCIPsetProbDeltrans(), SCIPsetProbInitsol(), SCIPsetProbExitsol(), and
 *  SCIPsetProbCopy()
 *  If the problem type requires the use of variable pricers, these pricers should be added to the problem with calls
 *  to SCIPactivatePricer(). These pricers are automatically deactivated, when the problem is freed.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_FREE
 *
 *  @post After calling this method, \SCIP reaches the following stage:
 *        - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPcreateProbBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< problem name */
   );

/** sets callback to free user data of original problem
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 */
SCIP_RETCODE SCIPsetProbDelorig(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DECL_PROBDELORIG ((*probdelorig))    /**< frees user data of original problem */
   );

/** sets callback to create user data of transformed problem by transforming original user data
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetProbTrans(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DECL_PROBTRANS   ((*probtrans))      /**< creates user data of transformed problem by transforming original user data */
   );

/** sets callback to free user data of transformed problem
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetProbDeltrans(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DECL_PROBDELTRANS((*probdeltrans))   /**< frees user data of transformed problem */
   );

/** sets solving process initialization callback of transformed data
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetProbInitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DECL_PROBINITSOL ((*probinitsol))    /**< solving process initialization method of transformed data */
   );

/** sets solving process deinitialization callback of transformed data
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetProbExitsol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DECL_PROBEXITSOL ((*probexitsol))    /**< solving process deinitialization method of transformed data */
   );

/** sets callback to copy user data to a subscip
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetProbCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DECL_PROBCOPY    ((*probcopy))       /**< copies user data if you want to copy it to a subscip, or NULL */
   );

/** reads problem from file and initializes all solving data structures
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @post After the method was called, \SCIP is in one of the following stages:
 *       -\ref SCIP_STAGE_INIT if reading failed (usually, when a SCIP_READERROR occurs)
 *       - ref SCIP_STAGE_PROBLEM if the problem file was successfully read
 */
EXTERN
SCIP_RETCODE SCIPreadProb(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< problem file name */
   const char*           extension           /**< extension of the desired file reader, 
                                              *   or NULL if file extension should be used */
   );

/** writes original problem to file
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPwriteOrigProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< output file (or NULL for standard output) */
   const char*           extension,          /**< extension of the desired file reader, 
                                              *   or NULL if file extension should be used */
   SCIP_Bool             genericnames        /**< use generic variable and constraint names? */
   );

/** writes transformed problem which are valid in the current node to file
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @note If you want the write all constraints (including the once which are redundant for example), you need to set
 *        the parameter <write/allconss> to TRUE
 */
EXTERN
SCIP_RETCODE SCIPwriteTransProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< output file (or NULL for standard output) */
   const char*           extension,          /**< extension of the desired file reader, 
                                              *   or NULL if file extension should be used */
   SCIP_Bool             genericnames        /**< using generic variable and constraint names? */
   );

/** frees problem and solution process data
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_FREE
 *
 *  @post After this method was called, SCIP is in the following stage:
 *       - \ref SCIP_STAGE_INIT
 */
EXTERN
SCIP_RETCODE SCIPfreeProb(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** permutes parts of the problem data structure
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 */
EXTERN
SCIP_RETCODE SCIPpermuteProb(
   SCIP*                 scip,               /**< SCIP data structure */
   unsigned int          randseed,           /**< seed value for random generator */
   SCIP_Bool             permuteconss,       /**< should the list of constraints in each constraint handler be permuted? */
   SCIP_Bool             permutebinvars,     /**< should the list of binary variables be permuted? */
   SCIP_Bool             permuteintvars,     /**< should the list of integer variables be permuted? */
   SCIP_Bool             permuteimplvars,    /**< should the list of implicit integer variables be permuted? */
   SCIP_Bool             permutecontvars     /**< should the list of continuous integer variables be permuted? */
   );

/** gets user problem data
 *
 *  @return a SCIP_PROBDATA pointer, or NULL if no problem data was allocated
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_PROBDATA* SCIPgetProbData(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets user problem data
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPsetProbData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROBDATA*        probdata            /**< user problem data to use */
   );

/** returns name of the current problem instance
 *
 *  @return name of the current problem instance
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
const char* SCIPgetProbName(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets name of the current problem instance
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPsetProbName(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name to be set */
   );

/** returns objective sense of original problem
 *
 *  @return objective sense of original problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_OBJSENSE SCIPgetObjsense(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets objective sense of problem
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPsetObjsense(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_OBJSENSE         objsense            /**< new objective sense */
   );

/** adds offset of objective function
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddObjoffset(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             addval              /**< value to add to objective offset */
   );

/** returns the objective offset of the original problem
 *
 *  @return the objective offset of the original problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetOrigObjoffset(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the objective scale of the original problem
 *
 *  @return the objective scale of the original problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetOrigObjscale(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the objective offset of the transformed problem
 *
 *  @return the objective offset of the transformed problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetTransObjoffset(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the objective scale of the transformed problem
 *
 *  @return the objective scale of the transformed problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetTransObjscale(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets limit on objective function, such that only solutions better than this limit are accepted
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetObjlimit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             objlimit            /**< new primal objective limit */
   );

/** returns current limit on objective function
 *
 *  @return the current objective limit of the original problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetObjlimit(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** informs SCIP, that the objective value is always integral in every feasible solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. otherwise a suitable error code is passed. see \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetObjIntegral(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether the objective value is known to be integral in every feasible solution
 *
 *  @return TRUE, if objective value is known to be always integral, otherwise FALSE
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Bool SCIPisObjIntegral(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the Euclidean norm of the objective function vector (available only for transformed problem)
 *
 *  @return the Euclidean norm of the transformed objective function vector
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Real SCIPgetObjNorm(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** adds variable to the problem
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to add */
   );

/** adds variable to the problem and uses it as pricing candidate to enter the LP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can only be called if @p scip is in stage \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddPricedVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to add */
   SCIP_Real             score               /**< pricing score of variable (the larger, the better the variable) */
   );

/** removes variable from the problem
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPdelVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to delete */
   SCIP_Bool*            deleted             /**< pointer to store whether variable was successfully marked to be deleted */
   );

/** gets variables of the problem along with the numbers of different variable types; data may become invalid after
 *  calls to SCIPchgVarType(), SCIPfixVar(), SCIPaggregateVars(), and SCIPmultiaggregateVar()
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @note Variables in the vars array are ordered: binaries first, then integers, implicit integers and continuous last.
 */
EXTERN
SCIP_RETCODE SCIPgetVarsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           vars,               /**< pointer to store variables array or NULL if not needed */
   int*                  nvars,              /**< pointer to store number of variables or NULL if not needed */
   int*                  nbinvars,           /**< pointer to store number of binary variables or NULL if not needed */
   int*                  nintvars,           /**< pointer to store number of integer variables or NULL if not needed */
   int*                  nimplvars,          /**< pointer to store number of implicit integral vars or NULL if not needed */
   int*                  ncontvars           /**< pointer to store number of continuous variables or NULL if not needed */
   );

/** gets array with active problem variables
 *
 *  @return array with active problem variables
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @warning If your are using the methods which add or change bound of variables (e.g., SCIPchgVarType(), SCIPfixVar(),
 *           SCIPaggregateVars(), and SCIPmultiaggregateVar()), it can happen that the internal variable array (which is
 *           accessed via this method) gets resized and/or resorted. This can invalid the data pointer which is returned
 *           by this method.
 *
 *  @note Variables in the array are ordered: binaries first, then integers, implicit integers and continuous last.
 */
EXTERN
SCIP_VAR** SCIPgetVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of active problem variables
 *
 *  @return the number of active problem variables
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetNVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of binary active problem variables
 *
 *  @return the number of binary active problem variables
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetNBinVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of integer active problem variables
 *
 *  @return the number of integer active problem variables
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetNIntVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of implicit integer active problem variables
 *
 *  @return the number of implicit integer active problem variables
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetNImplVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of continuous active problem variables
 *
 *  @return the number of continuous active problem variables
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetNContVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of active problem variables with a non-zero objective coefficient
 *
 *  @note In case of the original problem the number of variables is counted. In case of the transformed problem the
 *        number of variables is just returned since it is stored internally
 *
 *  @return the number of active problem variables with a non-zero objective coefficient
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNObjVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets array with fixed and aggregated problem variables; data may become invalid after
 *  calls to SCIPfixVar(), SCIPaggregateVars(), and SCIPmultiaggregateVar()
 *
 *  @return an array with fixed and aggregated problem variables; data may become invalid after
 *          calls to SCIPfixVar(), SCIPaggregateVars(), and SCIPmultiaggregateVar()
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_VAR** SCIPgetFixedVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of fixed or aggregated problem variables
 *
 *  @return the number of fixed or aggregated problem variables
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNFixedVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets variables of the original problem along with the numbers of different variable types; data may become invalid
 *  after a call to SCIPchgVarType()
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPgetOrigVarsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           vars,               /**< pointer to store variables array or NULL if not needed */
   int*                  nvars,              /**< pointer to store number of variables or NULL if not needed */
   int*                  nbinvars,           /**< pointer to store number of binary variables or NULL if not needed */
   int*                  nintvars,           /**< pointer to store number of integer variables or NULL if not needed */
   int*                  nimplvars,          /**< pointer to store number of implicit integral vars or NULL if not needed */
   int*                  ncontvars           /**< pointer to store number of continuous variables or NULL if not needed */
   );

/** gets array with original problem variables; data may become invalid after
 *  a call to SCIPchgVarType()
 *
 *  @return an array with original problem variables; data may become invalid after
 *          a call to SCIPchgVarType()
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_VAR** SCIPgetOrigVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of original problem variables
 *
 *  @return the number of original problem variables
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
int SCIPgetNOrigVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of binary variables in the original problem
 *
 *  @return the number of binary variables in the original problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
int SCIPgetNOrigBinVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the number of integer variables in the original problem
 *
 *  @return the number of integer variables in the original problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
int SCIPgetNOrigIntVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of implicit integer variables in the original problem
 *
 *  @return the number of implicit integer variables in the original problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
int SCIPgetNOrigImplVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of continuous variables in the original problem
 *
 *  @return the number of continuous variables in the original problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
int SCIPgetNOrigContVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of all problem variables created during creation and solving of problem;
 *  this includes also variables that were deleted in the meantime
 *
 *  @return the number of all problem variables created during creation and solving of problem;
 *          this includes also variables that were deleted in the meantime
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
int SCIPgetNTotalVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets variables of the original or transformed problem along with the numbers of different variable types;
 *  the returned problem space (original or transformed) corresponds to the given solution;
 *  data may become invalid after calls to SCIPchgVarType(), SCIPfixVar(), SCIPaggregateVars(), and 
 *  SCIPmultiaggregateVar()
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPgetSolVarsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution that selects the problem space, NULL for current solution */
   SCIP_VAR***           vars,               /**< pointer to store variables array or NULL if not needed */
   int*                  nvars,              /**< pointer to store number of variables or NULL if not needed */
   int*                  nbinvars,           /**< pointer to store number of binary variables or NULL if not needed */
   int*                  nintvars,           /**< pointer to store number of integer variables or NULL if not needed */
   int*                  nimplvars,          /**< pointer to store number of implicit integral vars or NULL if not needed */
   int*                  ncontvars           /**< pointer to store number of continuous variables or NULL if not needed */
   );

/** returns variable of given name in the problem, or NULL if not existing
 *
 *  @return variable of given name in the problem, or NULL if not existing
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_VAR* SCIPfindVar(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of variable to find */
   );

/** returns TRUE iff all potential variables exist in the problem, and FALSE, if there may be additional variables,
 *  that will be added in pricing and improve the objective value
 *
 *  @return TRUE, if all potential variables exist in the problem; FALSE, otherwise
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Bool SCIPallVarsInProb(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** adds constraint to the problem; if constraint is only valid locally, it is added to the local subproblem of the
 *  current node (and all of its subnodes); otherwise it is added to the global problem;
 *  if a local constraint is added at the root node, it is automatically upgraded into a global constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPaddCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint to add */
   );

/** globally removes constraint from all subproblems; removes constraint from the constraint set change data of the
 *  node, where it was added, or from the problem, if it was a problem constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPdelCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint to delete */
   );

/** returns original constraint of given name in the problem, or NULL if not existing
 *
 *  @return original constraint of given name in the problem, or NULL if not existing
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS */
EXTERN
SCIP_CONS* SCIPfindOrigCons(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of constraint to find */
   );

/** returns constraint of given name in the problem, or NULL if not existing
 *
 *  @return constraint of given name in the problem, or NULL if not existing
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_CONS* SCIPfindCons(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of constraint to find */
   );

/** gets number of upgraded constraints
 *
 *  @return number of upgraded constraints
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNUpgrConss(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of globally valid constraints currently in the problem
 *
 *  @return total number of globally valid constraints currently in the problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNConss(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets array of globally valid constraints currently in the problem
 *
 *  @return array of globally valid constraints currently in the problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @warning If your are using the method SCIPaddCons(), it can happen that the internal constraint array (which is
 *           accessed via this method) gets resized. This can invalid the pointer which is returned by this method.
 */
EXTERN
SCIP_CONS** SCIPgetConss(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of constraints in the original problem
 *
 *  @return total number of constraints in the original problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
int SCIPgetNOrigConss(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets array of constraints in the original problem
 *
 *  @return array of constraints in the original problem
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_CONS** SCIPgetOrigConss(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@} */




/*
 * local subproblem methods
 */

/**@name Local Subproblem Methods */
/**@{ */

/** adds constraint to the given node (and all of its subnodes), even if it is a global constraint;
 *  It is sometimes desirable to add the constraint to a more local node (i.e., a node of larger depth) even if
 *  the constraint is also valid higher in the tree, for example, if one wants to produce a constraint which is
 *  only active in a small part of the tree although it is valid in a larger part.
 *  In this case, one should pass the more global node where the constraint is valid as "validnode".
 *  Note that the same constraint cannot be added twice to the branching tree with different "validnode" parameters.
 *  If the constraint is valid at the same node as it is inserted (the usual case), one should pass NULL as "validnode".
 *  If the "validnode" is the root node, it is automatically upgraded into a global constraint, but still only added to
 *  the given node. If a local constraint is added to the root node, it is added to the global problem instead.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddConsNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to add constraint to */
   SCIP_CONS*            cons,               /**< constraint to add */
   SCIP_NODE*            validnode           /**< node at which the constraint is valid, or NULL */
   );

/** adds constraint locally to the current node (and all of its subnodes), even if it is a global constraint;
 *  It is sometimes desirable to add the constraint to a more local node (i.e., a node of larger depth) even if
 *  the constraint is also valid higher in the tree, for example, if one wants to produce a constraint which is
 *  only active in a small part of the tree although it is valid in a larger part.
 *
 *  If the constraint is valid at the same node as it is inserted (the usual case), one should pass NULL as "validnode".
 *  If the "validnode" is the root node, it is automatically upgraded into a global constraint, but still only added to
 *  the given node. If a local constraint is added to the root node, it is added to the global problem instead.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note The same constraint cannot be added twice to the branching tree with different "validnode" parameters. This is
 *        the case due internal data structures and performance issues. In such a case you should try to realize your
 *        issue using the method SCIPdisableCons() and SCIPenableCons() and control these via the event system of SCIP.
 */
EXTERN
SCIP_RETCODE SCIPaddConsLocal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to add */
   SCIP_NODE*            validnode           /**< node at which the constraint is valid, or NULL */
   );

/** disables constraint's separation, enforcing, and propagation capabilities at the given node (and all subnodes);
 *  if the method is called at the root node, the constraint is globally deleted from the problem;
 *  the constraint deletion is being remembered at the given node, s.t. after leaving the node's subtree, the constraint
 *  is automatically enabled again, and after entering the node's subtree, it is automatically disabled;
 *  this may improve performance because redundant checks on this constraint are avoided, but it consumes memory;
 *  alternatively, use SCIPdisableCons()
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPdelConsNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to disable constraint in */
   SCIP_CONS*            cons                /**< constraint to locally delete */
   );

/** disables constraint's separation, enforcing, and propagation capabilities at the current node (and all subnodes);
 *  if the method is called during problem modification or at the root node, the constraint is globally deleted from
 *  the problem;
 *  the constraint deletion is being remembered at the current node, s.t. after leaving the current subtree, the
 *  constraint is automatically enabled again, and after reentering the current node's subtree, it is automatically
 *  disabled again;
 *  this may improve performance because redundant checks on this constraint are avoided, but it consumes memory;
 *  alternatively, use SCIPdisableCons()
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPdelConsLocal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint to locally delete */
   );

/** gets estimate of best primal solution w.r.t. original problem contained in current subtree
 *
 *  @return estimate of best primal solution w.r.t. original problem contained in current subtree
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetLocalOrigEstimate(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets estimate of best primal solution w.r.t. transformed problem contained in current subtree
 *
 *  @return estimate of best primal solution w.r.t. transformed problem contained in current subtree
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetLocalTransEstimate(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets dual bound of current node
 *
 *  @return dual bound of current node
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetLocalDualbound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets lower bound of current node in transformed problem
 *
 *  @return lower bound  of current node in transformed problem
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetLocalLowerbound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets dual bound of given node
 *
 *  @return dual bound of a given node
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetNodeDualbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node                /**< node to get dual bound for */
   );

/** gets lower bound of given node in transformed problem
 *
 *  @return lower bound  of given node in transformed problem
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetNodeLowerbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node                /**< node to get dual bound for */
   );

/** if given value is tighter (larger for minimization, smaller for maximization) than the current node's dual bound (in
 *  original problem space), sets the current node's dual bound to the new value
 *
 *  @note the given new bound has to be a dual bound, i.e., it has to be valid for the original problem.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPupdateLocalDualbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             newbound            /**< new dual bound for the node (if it's tighter than the old one) */
   );

/** if given value is larger than the current node's lower bound (in transformed problem), sets the current node's
 *  lower bound to the new value
 *
 *  @note the given new bound has to be a lower bound, i.e., it has to be valid for the transformed problem.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPupdateLocalLowerbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             newbound            /**< new lower bound for the node (if it's larger than the old one) */
   );

/** if given value is tighter (larger for minimization, smaller for maximization) than the node's dual bound,
 *  sets the node's dual bound to the new value
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPupdateNodeDualbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to update dual bound for */
   SCIP_Real             newbound            /**< new dual bound for the node (if it's tighter than the old one) */
   );

/** if given value is larger than the node's lower bound (in transformed problem), sets the node's lower bound
 *  to the new value
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPupdateNodeLowerbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to update lower bound for */
   SCIP_Real             newbound            /**< new lower bound for the node (if it's larger than the old one) */
   );

/** change the node selection priority of the given child
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgChildPrio(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            child,              /**< child to update the node selection priority */
   SCIP_Real             priority            /**< node selection priority value */
   );

/**@} */




/*
 * solve methods
 */

/**@name Solve Methods */
/**@{ */

/** initializes solving data structures and transforms problem
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *       - \ref SCIP_STAGE_FREE
 *
 *  @post When calling this method in the \ref SCIP_STAGE_PROBLEM stage, the \SCIP stage is changed to \ref
 *        SCIP_STAGE_TRANSFORMED; otherwise, the stage is not changed
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPtransformProb(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** transforms and presolves problem
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *
 *  @post After calling this method \SCIP reaches one of the following stages:
 *        - \ref SCIP_STAGE_PRESOLVING if the presolving process was interrupted
 *        - \ref SCIP_STAGE_PRESOLVED if the presolving process was finished and did not solve the problem
 *        - \ref SCIP_STAGE_SOLVED if the problem was solved during presolving
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPpresolve(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** transforms, presolves, and solves problem
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @post After calling this method \SCIP reaches one of the following stages depending on if and when the solution
 *        process was interrupted:

 *        - \ref SCIP_STAGE_PRESOLVING if the solution process was interrupted during presolving
 *        - \ref SCIP_STAGE_SOLVING if the solution process was interrupted during the tree search
 *        - \ref SCIP_STAGE_SOLVED if the solving process was not interrupted
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPsolve(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** frees branch and bound tree and all solution process data; statistics, presolving data and transformed problem is
 *  preserved
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @post If this method is called in \SCIP stage \ref SCIP_STAGE_INIT or \ref SCIP_STAGE_PROBLEM, the stage of
 *        \SCIP is not changed; otherwise, the \SCIP stage is changed to \ref SCIP_STAGE_TRANSFORMED
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPfreeSolve(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             restart             /**< should certain data be preserved for improved restarting? */
   );

/** frees all solution process data including presolving and transformed problem, only original problem is kept
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @post After calling this method \SCIP reaches one of the following stages:
 *        - \ref SCIP_STAGE_INIT if the method was called from \SCIP stage \ref SCIP_STAGE_INIT
 *        - \ref SCIP_STAGE_PROBLEM if the method was called from any other of the allowed stages
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPfreeTransform(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** informs \SCIP that the solving process should be interrupted as soon as possible (e.g., after the current node has
 *   been solved)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note the \SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPinterruptSolve(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** informs SCIP that the solving process should be restarted as soon as possible (e.g., after the current node has
 *  been solved)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note the \SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPrestartSolve(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether we are in the restarting phase
 *
 *  @return TRUE, if we are in the restarting phase; FALSE, otherwise
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Bool SCIPisInRestart(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@} */




/*
 * variable methods
 */

/**@name Variable Methods */
/**@{ */

/** creates and captures problem variable; if variable is of integral type, fractional bounds are automatically rounded;
 *  an integer variable with bounds zero and one is automatically converted into a binary variable;
 *
 *  @warning When doing column generation and the original problem is a maximization problem, notice that SCIP will
 *           transform the problem into a minimization problem by multiplying the objective function by -1.  Thus, the
 *           original objective function value of variables created during the solving process has to be multiplied by
 *           -1, too.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note the variable gets captured, hence at one point you have to release it using the method SCIPreleaseVar()
 */
EXTERN
SCIP_RETCODE SCIPcreateVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            var,                /**< pointer to variable object */
   const char*           name,               /**< name of variable, or NULL for automatic name creation */
   SCIP_Real             lb,                 /**< lower bound of variable */
   SCIP_Real             ub,                 /**< upper bound of variable */
   SCIP_Real             obj,                /**< objective function value */
   SCIP_VARTYPE          vartype,            /**< type of variable */
   SCIP_Bool             initial,            /**< should var's column be present in the initial root LP? */
   SCIP_Bool             removable,          /**< is var's column removable from the LP (due to aging or cleanup)? */
   SCIP_DECL_VARDELORIG  ((*vardelorig)),    /**< frees user data of original variable, or NULL */
   SCIP_DECL_VARTRANS    ((*vartrans)),      /**< creates transformed user data by transforming original user data, or NULL */
   SCIP_DECL_VARDELTRANS ((*vardeltrans)),   /**< frees user data of transformed variable, or NULL */
   SCIP_DECL_VARCOPY     ((*varcopy)),       /**< copies variable data if wanted to subscip, or NULL */
   SCIP_VARDATA*         vardata             /**< user data for this specific variable, or NULL */
   );

/** creates and captures problem variable with optional callbacks and variable data set to NULL, which can be set
 *  afterwards using SCIPvarSetDelorigData(), SCIPvarSetTransData(),
 *  SCIPvarSetDeltransData(), SCIPvarSetCopy(), and SCIPvarSetData(); sets variable flags initial=TRUE
 *  and removable = FALSE, which can be adjusted by using SCIPvarSetInitial() and SCIPvarSetRemovable(), resp.;
 *  if variable is of integral type, fractional bounds are automatically rounded;
 *  an integer variable with bounds zero and one is automatically converted into a binary variable;
 *
 *  @warning When doing column generation and the original problem is a maximization problem, notice that SCIP will
 *           transform the problem into a minimization problem by multiplying the objective function by -1.  Thus, the
 *           original objective function value of variables created during the solving process has to be multiplied by
 *           -1, too.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note the variable gets captured, hence at one point you have to release it using the method SCIPreleaseVar()
 */
EXTERN
SCIP_RETCODE SCIPcreateVarBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            var,                /**< pointer to variable object */
   const char*           name,               /**< name of variable, or NULL for automatic name creation */
   SCIP_Real             lb,                 /**< lower bound of variable */
   SCIP_Real             ub,                 /**< upper bound of variable */
   SCIP_Real             obj,                /**< objective function value */
   SCIP_VARTYPE          vartype             /**< type of variable */
   );

/** outputs the variable name to the file stream
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPwriteVarName(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file, or NULL for stdout */
   SCIP_VAR*             var,                /**< variable to output */
   SCIP_Bool             type                /**< should the variable type be also posted */
   );

/** print the given list of variables to output stream separated by the given delimiter character;
 *
 *  i. e. the variables x1, x2, ..., xn with given delimiter ',' are written as: \<x1\>, \<x2\>, ..., \<xn\>;
 *
 *  the method SCIPparseVarsList() can parse such a string
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note The printing process is done via the message handler system.
 */
EXTERN
SCIP_RETCODE SCIPwriteVarsList(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file, or NULL for stdout */
   SCIP_VAR**            vars,               /**< variable array to output */
   int                   nvars,              /**< number of variables */
   SCIP_Bool             type,               /**< should the variable type be also posted */
   char                  delimiter           /**< character which is used for delimitation */
   );

/** print the given variables and coefficients as linear sum in the following form
 *  c1 \<x1\> + c2 \<x2\>   ... + cn \<xn\>
 *
 *  This string can be parsed by the method SCIPparseVarsLinearsum().
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note The printing process is done via the message handler system.
 */
EXTERN
SCIP_RETCODE SCIPwriteVarsLinearsum(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file, or NULL for stdout */
   SCIP_VAR**            vars,               /**< variable array to output */
   SCIP_Real*            vals,               /**< array of coefficients or NULL if all coefficients are 1.0 */
   int                   nvars,              /**< number of variables */
   SCIP_Bool             type                /**< should the variable type be also posted */
   );

/** print the given monomials as polynomial in the following form
 *  c1 \<x11\>^e11 \<x12\>^e12 ... \<x1n\>^e1n + c2 \<x21\>^e21 \<x22\>^e22 ... + ... + cn \<xn1\>^en1 ...
 *
 *  This string can be parsed by the method SCIPparseVarsPolynomial().
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note The printing process is done via the message handler system.
 */
EXTERN
SCIP_RETCODE SCIPwriteVarsPolynomial(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file, or NULL for stdout */
   SCIP_VAR***           monomialvars,       /**< arrays with variables for each monomial */
   SCIP_Real**           monomialexps,       /**< arrays with variable exponents, or NULL if always 1.0 */
   SCIP_Real*            monomialcoefs,      /**< array with monomial coefficients */
   int*                  monomialnvars,      /**< array with number of variables for each monomial */
   int                   nmonomials,         /**< number of monomials */
   SCIP_Bool             type                /**< should the variable type be also posted */
   );

/** parses variable information (in cip format) out of a string; if the parsing process was successful a variable is
 *  created and captured; if variable is of integral type, fractional bounds are automatically rounded; an integer
 *  variable with bounds zero and one is automatically converted into a binary variable
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPparseVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            var,                /**< pointer to store the problem variable */
   const char*           str,                /**< string to parse */
   SCIP_Bool             initial,            /**< should var's column be present in the initial root LP? */
   SCIP_Bool             removable,          /**< is var's column removable from the LP (due to aging or cleanup)? */
   SCIP_DECL_VARCOPY     ((*varcopy)),       /**< copies variable data if wanted to subscip, or NULL */
   SCIP_DECL_VARDELORIG  ((*vardelorig)),    /**< frees user data of original variable */
   SCIP_DECL_VARTRANS    ((*vartrans)),      /**< creates transformed user data by transforming original user data */
   SCIP_DECL_VARDELTRANS ((*vardeltrans)),   /**< frees user data of transformed variable */
   SCIP_VARDATA*         vardata,            /**< user data for this specific variable */
   char**                endptr,             /**< pointer to store the final string position if successfully */
   SCIP_Bool*            success             /**< pointer store if the paring process was successful */
   );

/** parses the given string for a variable name and stores the variable in the corresponding pointer if such a variable
 *  exits and returns the position where the parsing stopped 
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPparseVarName(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           str,                /**< string to parse */
   SCIP_VAR**            var,                /**< pointer to store the problem variable, or NULL if it does not exit */
   char**                endptr              /**< pointer to store the final string position if successfully */
   );

/** parse the given string as variable list (here ',' is the delimiter)) (\<x1\>, \<x2\>, ..., \<xn\>) (see
 *  SCIPwriteVarsList() ); if it was successful, the pointer success is set to TRUE
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note the pointer success in only set to FALSE in the case that a variable with a parsed variable name does not exist
 *
 *  @note If the number of (parsed) variables is greater than the available slots in the variable array, nothing happens
 *        except that the required size is stored in the corresponding integer; the reason for this approach is that we
 *        cannot reallocate memory, since we do not know how the memory has been allocated (e.g., by a C++ 'new' or SCIP
 *        memory functions).
 */
EXTERN
SCIP_RETCODE SCIPparseVarsList(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           str,                /**< string to parse */
   SCIP_VAR**            vars,               /**< array to store the parsed variable */
   int*                  nvars,              /**< pointer to store number of parsed variables */
   int                   varssize,           /**< size of the variable array */
   int*                  requiredsize,       /**< pointer to store the required array size for the active variables */
   char**                endptr,             /**< pointer to store the final string position if successfully */
   char                  delimiter,          /**< character which is used for delimitation */
   SCIP_Bool*            success             /**< pointer to store the whether the parsing was successfully or not */
   );

/** parse the given string as linear sum of variables and coefficients (c1 \<x1\> + c2 \<x2\> + ... + cn \<xn\>)
 *  (see SCIPwriteVarsLinearsum() ); if it was successful, the pointer success is set to TRUE
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note the pointer success in only set to FALSE in the case that a variable with a parsed variable name does not exist
 *
 *  @note If the number of (parsed) variables is greater than the available slots in the variable array, nothing happens
 *        except that the required size is stored in the corresponding integer; the reason for this approach is that we
 *        cannot reallocate memory, since we do not know how the memory has been allocated (e.g., by a C++ 'new' or SCIP
 *        memory functions).
 */
EXTERN
SCIP_RETCODE SCIPparseVarsLinearsum(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           str,                /**< string to parse */
   SCIP_VAR**            vars,               /**< array to store the parsed variables */
   SCIP_Real*            vals,               /**< array to store the parsed coefficients */
   int*                  nvars,              /**< pointer to store number of parsed variables */
   int                   varssize,           /**< size of the variable array */
   int*                  requiredsize,       /**< pointer to store the required array size for the active variables */
   char**                endptr,             /**< pointer to store the final string position if successfully */
   SCIP_Bool*            success             /**< pointer to store the whether the parsing was successfully or not */
   );

/** parse the given string as polynomial of variables and coefficients
 *  (c1 \<x11\>^e11 \<x12\>^e12 ... \<x1n\>^e1n + c2 \<x21\>^e21 \<x22\>^e22 ... + ... + cn \<xn1\>^en1 ...)
 *  (see SCIPwriteVarsPolynomial()); if it was successful, the pointer success is set to TRUE
 *
 *  The user has to call SCIPfreeParseVarsPolynomialData(scip, monomialvars, monomialexps,
 *  monomialcoefs, monomialnvars, *nmonomials) short after SCIPparseVarsPolynomial to free all the
 *  allocated memory again.  Do not keep the arrays created by SCIPparseVarsPolynomial around, since
 *  they use buffer memory that is intended for short term use only.
 *
 *  Parsing is stopped at the end of string (indicated by the \\0-character) or when no more monomials
 *  are recognized.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPparseVarsPolynomial(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           str,                /**< string to parse */
   SCIP_VAR****          monomialvars,       /**< pointer to store arrays with variables for each monomial */
   SCIP_Real***          monomialexps,       /**< pointer to store arrays with variable exponents */
   SCIP_Real**           monomialcoefs,      /**< pointer to store array with monomial coefficients */
   int**                 monomialnvars,      /**< pointer to store array with number of variables for each monomial */
   int*                  nmonomials,         /**< pointer to store number of parsed monomials */
   char**                endptr,             /**< pointer to store the final string position if successfully */
   SCIP_Bool*            success             /**< pointer to store the whether the parsing was successfully or not */
   );

/** frees memory allocated when parsing a polynomial from a string
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
void SCIPfreeParseVarsPolynomialData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR****          monomialvars,       /**< pointer to store arrays with variables for each monomial */
   SCIP_Real***          monomialexps,       /**< pointer to store arrays with variable exponents */
   SCIP_Real**           monomialcoefs,      /**< pointer to store array with monomial coefficients */
   int**                 monomialnvars,      /**< pointer to store array with number of variables for each monomial */
   int                   nmonomials          /**< pointer to store number of parsed monomials */
   );

/** increases usage counter of variable
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPcaptureVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to capture */
   );

/** decreases usage counter of variable, if the usage pointer reaches zero the variable gets freed
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note the pointer of the variable will be NULLed
 */
EXTERN
SCIP_RETCODE SCIPreleaseVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            var                 /**< pointer to variable */
   );

/** changes the name of a variable
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can only be called if @p scip is in stage \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPchgVarName(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable */
   const char*           name                /**< new name of constraint */
   );

/** gets and captures transformed variable of a given variable; if the variable is not yet transformed,
 *  a new transformed variable for this variable is created
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPtransformVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get/create transformed variable for */
   SCIP_VAR**            transvar            /**< pointer to store the transformed variable */
   );

/** gets and captures transformed variables for an array of variables;
 *  if a variable of the array is not yet transformed, a new transformed variable for this variable is created;
 *  it is possible to call this method with vars == transvars
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPtransformVars(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nvars,              /**< number of variables to get/create transformed variables for */
   SCIP_VAR**            vars,               /**< array with variables to get/create transformed variables for */
   SCIP_VAR**            transvars           /**< array to store the transformed variables */
   );

/** gets corresponding transformed variable of a given variable;
 *  returns NULL as transvar, if transformed variable is not yet existing
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPgetTransformedVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get transformed variable for */
   SCIP_VAR**            transvar            /**< pointer to store the transformed variable */
   );

/** gets corresponding transformed variables for an array of variables;
 *  stores NULL in a transvars slot, if the transformed variable is not yet existing;
 *  it is possible to call this method with vars == transvars, but remember that variables that are not
 *  yet transformed will be replaced with NULL
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPgetTransformedVars(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nvars,              /**< number of variables to get transformed variables for */
   SCIP_VAR**            vars,               /**< array with variables to get transformed variables for */
   SCIP_VAR**            transvars           /**< array to store the transformed variables */
   );

/** gets negated variable x' = lb + ub - x of variable x; negated variable is created, if not yet existing
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPgetNegatedVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get negated variable for */
   SCIP_VAR**            negvar              /**< pointer to store the negated variable */
   );

/** gets negated variables x' = lb + ub - x of variables x; negated variables are created, if not yet existing
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPgetNegatedVars(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nvars,              /**< number of variables to get negated variables for */
   SCIP_VAR**            vars,               /**< array of variables to get negated variables for */
   SCIP_VAR**            negvars             /**< array to store the negated variables */
   );

/** gets a binary variable that is equal to the given binary variable, and that is either active, fixed, or
 *  multi-aggregated, or the negated variable of an active, fixed, or multi-aggregated variable
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPgetBinvarRepresentative(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< binary variable to get binary representative for */
   SCIP_VAR**            repvar,             /**< pointer to store the binary representative */
   SCIP_Bool*            negated             /**< pointer to store whether the negation of an active variable was returned */
   );

/** gets binary variables that are equal to the given binary variables, and which are either active, fixed, or 
 *  multi-aggregated, or the negated variables of active, fixed, or multi-aggregated variables
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPgetBinvarRepresentatives(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nvars,              /**< number of binary variables to get representatives for */
   SCIP_VAR**            vars,               /**< binary variables to get binary representatives for */
   SCIP_VAR**            repvars,            /**< array to store the binary representatives */
   SCIP_Bool*            negated             /**< array to store whether the negation of an active variable was returned */
   );

/** flattens aggregation graph of multi-aggregated variable in order to avoid exponential recursion later on
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPflattenVarAggregationGraph(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** Transforms a given linear sum of variables, that is a_1*x_1 + ... + a_n*x_n + c into a corresponding linear sum of
 *  active variables, that is b_1*y_1 + ... + b_m*y_m + d.
 *
 *  If the number of needed active variables is greater than the available slots in the variable array, nothing happens
 *  except that the required size is stored in the corresponding variable (requiredsize). Otherwise, the active variable
 *  representation is stored in the variable array, scalar array and constant.
 *
 *  The reason for this approach is that we cannot reallocate memory, since we do not know how the memory has been
 *  allocated (e.g., by a C++ 'new' or SCIP functions).
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note The resulting linear sum is stored into the given variable array, scalar array, and constant. That means the
 *        given entries are overwritten.
 *
 *  @note That method can be used to convert a single variables into variable space of active variables. Therefore call
 *        the method with the linear sum 1.0*x + 0.0.
 */
EXTERN
SCIP_RETCODE SCIPgetProbvarLinearSum(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            vars,               /**< variable array x_1, ..., x_n in the linear sum which will be
                                              *   overwritten by the variable array y_1, ..., y_m in the linear sum
                                              *   w.r.t. active variables */
   SCIP_Real*            scalars,            /**< scalars a_1, ..., a_n in linear sum which will be overwritten to the
                                              *   scalars b_1, ..., b_m in the linear sum of the active variables  */
   int*                  nvars,              /**< pointer to number of variables in the linear sum which will be
                                              *   overwritten by the number of variables in the linear sum corresponding
                                              *   to the active variables */
   int                   varssize,           /**< available slots in vars and scalars array which is needed to check if
                                              *   the array are large enough for the linear sum w.r.t. active
                                              *   variables */
   SCIP_Real*            constant,           /**< pointer to constant c in linear sum a_1*x_1 + ... + a_n*x_n + c which
                                              *   will chnage to constant d in the linear sum b_1*y_1 + ... + b_m*y_m +
                                              *   d w.r.t. the active variables */
   int*                  requiredsize,       /**< pointer to store the required array size for the linear sum w.r.t. the
                                              *   active variables */
   SCIP_Bool             mergemultiples      /**< should multiple occurrences of a var be replaced by a single coeff? */
   );

/** transforms given variable, scalar and constant to the corresponding active, fixed, or
 *  multi-aggregated variable, scalar and constant; if the variable resolves to a fixed variable,
 *  "scalar" will be 0.0 and the value of the sum will be stored in "constant"
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPgetProbvarSum(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            var,                /**< pointer to problem variable x in sum a*x + c */
   SCIP_Real*            scalar,             /**< pointer to scalar a in sum a*x + c */
   SCIP_Real*            constant            /**< pointer to constant c in sum a*x + c */
   );

/** return for given variables all their active counterparts; all active variables will be pairwise different
 *  @note It does not hold that the first output variable is the active variable for the first input variable.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPgetActiveVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            vars,               /**< variable array with given variables and as output all active
                                              *   variables, if enough slots exist */
   int*                  nvars,              /**< number of given variables, and as output number of active variables,
                                              *   if enough slots exist */
   int                   varssize,           /**< available slots in vars array */
   int*                  requiredsize        /**< pointer to store the required array size for the active variables */
   );

/** returns the reduced costs of the variable in the current node's LP relaxation;
 *  the current node has to have a feasible LP.
 *
 *  returns SCIP_INVALID if the variable is active but not in the current LP;
 *  returns 0 if the variable has been aggregated out or fixed in presolving.
 *
 *  @pre This method can only be called if @p scip is in stage \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetVarRedcost(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get reduced costs, should be a column in current node LP */
   );

/** returns the implied reduced costs of the variable in the current node's LP relaxation;
 *  the current node has to have a feasible LP.
 *
 *  returns SCIP_INVALID if the variable is active but not in the current LP;
 *  returns 0 if the variable has been aggregated out or fixed in presolving.
 *
 *  @pre This method can only be called if @p scip is in stage \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetVarImplRedcost(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get reduced costs, should be a column in current node LP */
   SCIP_Bool             varfixing           /**< FALSE if for x == 0, TRUE for x == 1 */
   );

/** returns the Farkas coefficient of the variable in the current node's LP relaxation;
 *  the current node has to have an infeasible LP.
 *
 *  returns SCIP_INVALID if the variable is active but not in the current LP;
 *  returns 0 if the variable has been aggregated out or fixed in presolving.
 *
 *  @pre This method can only be called if @p scip is in stage \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetVarFarkasCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get reduced costs, should be a column in current node LP */
   );

/** gets solution value for variable in current node
 *
 *  @return solution value for variable in current node
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetVarSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get solution value for */
   );

/** gets solution values of multiple variables in current node
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetVarSols(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nvars,              /**< number of variables to get solution value for */
   SCIP_VAR**            vars,               /**< array with variables to get value for */
   SCIP_Real*            vals                /**< array to store solution values of variables */
   );

/** sets the solution value of all variables in the global relaxation solution to zero
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPclearRelaxSolVals(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the value of the given variable in the global relaxation solution;
 *  this solution can be filled by the relaxation handlers  and can be used by heuristics and for separation;
 *  You can use SCIPclearRelaxSolVals() to set all values to zero, initially;
 *  after setting all solution values, you have to call SCIPmarkRelaxSolValid() 
 *  to inform SCIP that the stored solution is valid
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetRelaxSolVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to set value for */
   SCIP_Real             val                 /**< solution value of variable */
   );

/** sets the values of the given variables in the global relaxation solution;
 *  this solution can be filled by the relaxation handlers  and can be used by heuristics and for separation;
 *  the solution is automatically cleared, s.t. all other variables get value 0.0
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetRelaxSolVals(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nvars,              /**< number of variables to set relaxation solution value for */
   SCIP_VAR**            vars,               /**< array with variables to set value for */
   SCIP_Real*            vals                /**< array with solution values of variables */
   );

/** sets the values of the variables in the global relaxation solution to the values 
 *  in the given primal solution; the relaxation solution can be filled by the relaxation hanlders
 *  and might be used by heuristics and for separation
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetRelaxSolValsSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal relaxation solution */ 
   );

/** returns whether the relaxation solution is valid
 *
 *  @return TRUE, if the relaxation solution is valid; FALSE, otherwise
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Bool SCIPisRelaxSolValid(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** informs SCIP, that the relaxation solution is valid
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPmarkRelaxSolValid(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** informs SCIP, that the relaxation solution is invalid
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPmarkRelaxSolInvalid(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the relaxation solution value of the given variable
 *
 *  @return the relaxation solution value of the given variable
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRelaxSolVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get value for */
   );

/** gets the relaxation solution objective value
 *
 *  @return the objective value of the relaxation solution
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRelaxSolObj(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** start strong branching - call before any strong branching
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note if propagation is enabled, strong branching is not done directly on the LP, but probing nodes are created
 *        which allow to perform propagation but also creates some overhead
 */
EXTERN
SCIP_RETCODE SCIPstartStrongbranch(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             enablepropagation   /**< should propagation be done before solving the strong branching LP? */
   );

/** end strong branching - call after any strong branching
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPendStrongbranch(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets strong branching information on column variable with fractional value
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetVarStrongbranchFrac(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get strong branching values for */
   int                   itlim,              /**< iteration limit for strong branchings */
   SCIP_Real*            down,               /**< stores dual bound after branching column down */
   SCIP_Real*            up,                 /**< stores dual bound after branching column up */
   SCIP_Bool*            downvalid,          /**< stores whether the returned down value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            upvalid,            /**< stores whether the returned up value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            downinf,            /**< pointer to store whether the downwards branch is infeasible, or NULL */
   SCIP_Bool*            upinf,              /**< pointer to store whether the upwards branch is infeasible, or NULL */
   SCIP_Bool*            downconflict,       /**< pointer to store whether a conflict constraint was created for an
                                              *   infeasible downwards branch, or NULL */
   SCIP_Bool*            upconflict,         /**< pointer to store whether a conflict constraint was created for an
                                              *   infeasible upwards branch, or NULL */
   SCIP_Bool*            lperror             /**< pointer to store whether an unresolved LP error occurred or the
                                              *   solving process should be stopped (e.g., due to a time limit) */
   );

/** gets strong branching information with previous propagation on column variable
 *
 *  Before calling this method, the strong branching mode must have been activated by calling SCIPstartStrongbranch();
 *  after strong branching was done for all candidate variables, the strong branching mode must be ended by
 *  SCIPendStrongbranch(). Since this method applies domain propagation before strongbranching, propagation has to be be
 *  enabled in the SCIPstartStrongbranch() call.
 *
 *  Before solving the strong branching LP, domain propagation can be performed. The number of propagation rounds
 *  can be specified by the parameter @p maxproprounds.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetVarStrongbranchWithPropagation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get strong branching values for */
   SCIP_Real             solval,             /**< value of the variable in the current LP solution */
   SCIP_Real             lpobjval,           /**< LP objective value of the current LP solution */
   int                   itlim,              /**< iteration limit for strong branchings */
   int                   maxproprounds,      /**< maximum number of propagation rounds (-1: no limit, -2: parameter
                                              *   settings) */
   SCIP_Real*            down,               /**< stores dual bound after branching column down */
   SCIP_Real*            up,                 /**< stores dual bound after branching column up */
   SCIP_Bool*            downvalid,          /**< stores whether the returned down value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            upvalid,            /**< stores whether the returned up value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            downinf,            /**< pointer to store whether the downwards branch is infeasible, or NULL */
   SCIP_Bool*            upinf,              /**< pointer to store whether the upwards branch is infeasible, or NULL */
   SCIP_Bool*            downconflict,       /**< pointer to store whether a conflict constraint was created for an
                                              *   infeasible downwards branch, or NULL */
   SCIP_Bool*            upconflict,         /**< pointer to store whether a conflict constraint was created for an
                                              *   infeasible upwards branch, or NULL */
   SCIP_Bool*            lperror,            /**< pointer to store whether an unresolved LP error occurred or the
                                              *   solving process should be stopped (e.g., due to a time limit) */
   SCIP_Real*            newlbs,             /**< array to store valid lower bounds for the variables */
   SCIP_Real*            newubs              /**< array to store valid upper bounds for the variables */
   );

/** gets strong branching information on column variable x with integral LP solution value (val); that is, the down branch
 *  is (val -1.0) and the up brach ins (val +1.0)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note If the integral LP solution value is the lower or upper bound of the variable, the corresponding branch will be
 *        marked as infeasible. That is, the valid pointer and the infeasible pointer are set to TRUE.
 */
EXTERN
SCIP_RETCODE SCIPgetVarStrongbranchInt(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get strong branching values for */
   int                   itlim,              /**< iteration limit for strong branchings */
   SCIP_Real*            down,               /**< stores dual bound after branching column down */
   SCIP_Real*            up,                 /**< stores dual bound after branching column up */
   SCIP_Bool*            downvalid,          /**< stores whether the returned down value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            upvalid,            /**< stores whether the returned up value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            downinf,            /**< pointer to store whether the downwards branch is infeasible, or NULL */
   SCIP_Bool*            upinf,              /**< pointer to store whether the upwards branch is infeasible, or NULL */
   SCIP_Bool*            downconflict,       /**< pointer to store whether a conflict constraint was created for an
                                              *   infeasible downwards branch, or NULL */
   SCIP_Bool*            upconflict,         /**< pointer to store whether a conflict constraint was created for an
                                              *   infeasible upwards branch, or NULL */
   SCIP_Bool*            lperror             /**< pointer to store whether an unresolved LP error occurred or the
                                              *   solving process should be stopped (e.g., due to a time limit) */
   );

/** gets strong branching information on column variables with fractional values
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetVarsStrongbranchesFrac(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            vars,               /**< variables to get strong branching values for */
   int                   nvars,              /**< number of variables */
   int                   itlim,              /**< iteration limit for strong branchings */
   SCIP_Real*            down,               /**< stores dual bounds after branching variables down */
   SCIP_Real*            up,                 /**< stores dual bounds after branching variables up */
   SCIP_Bool*            downvalid,          /**< stores whether the returned down values are valid dual bounds, or NULL;
                                              *   otherwise, they can only be used as an estimate value */
   SCIP_Bool*            upvalid,            /**< stores whether the returned up values are valid dual bounds, or NULL;
                                              *   otherwise, they can only be used as an estimate value */
   SCIP_Bool*            downinf,            /**< array to store whether the downward branches are infeasible, or NULL */
   SCIP_Bool*            upinf,              /**< array to store whether the upward branches are infeasible, or NULL */
   SCIP_Bool*            downconflict,       /**< array to store whether conflict constraints were created for
                                              *   infeasible downward branches, or NULL */
   SCIP_Bool*            upconflict,         /**< array to store whether conflict constraints were created for
                                              *   infeasible upward branches, or NULL */
   SCIP_Bool*            lperror             /**< pointer to store whether an unresolved LP error occurred or the
                                              *   solving process should be stopped (e.g., due to a time limit) */
   );

/** gets strong branching information on column variables with integral values
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetVarsStrongbranchesInt(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            vars,               /**< variables to get strong branching values for */
   int                   nvars,              /**< number of variables */
   int                   itlim,              /**< iteration limit for strong branchings */
   SCIP_Real*            down,               /**< stores dual bounds after branching variables down */
   SCIP_Real*            up,                 /**< stores dual bounds after branching variables up */
   SCIP_Bool*            downvalid,          /**< stores whether the returned down values are valid dual bounds, or NULL;
                                              *   otherwise, they can only be used as an estimate value */
   SCIP_Bool*            upvalid,            /**< stores whether the returned up values are valid dual bounds, or NULL;
                                              *   otherwise, they can only be used as an estimate value */
   SCIP_Bool*            downinf,            /**< array to store whether the downward branches are infeasible, or NULL */
   SCIP_Bool*            upinf,              /**< array to store whether the upward branches are infeasible, or NULL */
   SCIP_Bool*            downconflict,       /**< array to store whether conflict constraints were created for
                                              *   infeasible downward branches, or NULL */
   SCIP_Bool*            upconflict,         /**< array to store whether conflict constraints were created for
                                              *   infeasible upward branches, or NULL */
   SCIP_Bool*            lperror             /**< pointer to store whether an unresolved LP error occurred or the
                                              *   solving process should be stopped (e.g., due to a time limit) */
   );

/** gets strong branching information on COLUMN variable of the last SCIPgetVarStrongbranch() call;
 *  returns values of SCIP_INVALID, if strong branching was not yet called on the given variable;
 *  keep in mind, that the returned old values may have nothing to do with the current LP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPgetVarStrongbranchLast(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get last strong branching values for */
   SCIP_Real*            down,               /**< stores dual bound after branching column down, or NULL */
   SCIP_Real*            up,                 /**< stores dual bound after branching column up, or NULL */
   SCIP_Bool*            downvalid,          /**< stores whether the returned down value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            upvalid,            /**< stores whether the returned up value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Real*            solval,             /**< stores LP solution value of variable at last strong branching call, or NULL */
   SCIP_Real*            lpobjval            /**< stores LP objective value at last strong branching call, or NULL */
   );

/** gets node number of the last node in current branch and bound run, where strong branching was used on the
 *  given variable, or -1 if strong branching was never applied to the variable in current run
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Longint SCIPgetVarStrongbranchNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get last strong branching node for */
   );

/** if strong branching was already applied on the variable at the current node, returns the number of LPs solved after
 *  the LP where the strong branching on this variable was applied;
 *  if strong branching was not yet applied on the variable at the current node, returns INT_MAX
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Longint SCIPgetVarStrongbranchLPAge(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get strong branching LP age for */
   );

/** gets number of times, strong branching was applied in current run on the given variable
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetVarNStrongbranchs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get last strong branching node for */
   );

/** adds given values to lock numbers of variable for rounding
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPaddVarLocks(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   int                   nlocksdown,         /**< modification in number of rounding down locks */
   int                   nlocksup            /**< modification in number of rounding up locks */
   );

/** locks rounding of variable with respect to the lock status of the constraint and its negation;
 *  this method should be called whenever the lock status of a variable in a constraint changes, for example if
 *  the coefficient of the variable changed its sign or if the left or right hand sides of the constraint were
 *  added or removed
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPlockVarCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             lockdown,           /**< should the rounding be locked in downwards direction? */
   SCIP_Bool             lockup              /**< should the rounding be locked in upwards direction? */
   );

/** unlocks rounding of variable with respect to the lock status of the constraint and its negation;
 *  this method should be called whenever the lock status of a variable in a constraint changes, for example if
 *  the coefficient of the variable changed its sign or if the left or right hand sides of the constraint were
 *  added or removed
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPunlockVarCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             lockdown,           /**< should the rounding be locked in downwards direction? */
   SCIP_Bool             lockup              /**< should the rounding be locked in upwards direction? */
   );

/** changes variable's objective value
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgVarObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the objective value for */
   SCIP_Real             newobj              /**< new objective value */
   );

/** adds value to variable's objective value
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddVarObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the objective value for */
   SCIP_Real             addobj              /**< additional objective value */
   );

/** returns the adjusted (i.e. rounded, if the given variable is of integral type) lower bound value;
 *  does not change the bounds of the variable
 *
 *  @return adjusted lower bound for the given variable; the bound of the variable is not changed
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Real SCIPadjustedVarLb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to adjust the bound for */
   SCIP_Real             lb                  /**< lower bound value to adjust */
   );

/** returns the adjusted (i.e. rounded, if the given variable is of integral type) upper bound value;
 *  does not change the bounds of the variable
 *
 *  @return adjusted upper bound for the given variable; the bound of the variable is not changed
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Real SCIPadjustedVarUb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to adjust the bound for */
   SCIP_Real             ub                  /**< upper bound value to adjust */
   );

/** depending on SCIP's stage, changes lower bound of variable in the problem, in preprocessing, or in current node;
 *  if possible, adjusts bound to integral value; doesn't store any inference information in the bound change, such
 *  that in conflict analysis, this change is treated like a branching decision
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPchgVarLb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   );

/** depending on SCIP's stage, changes upper bound of variable in the problem, in preprocessing, or in current node;
 *  if possible, adjusts bound to integral value; doesn't store any inference information in the bound change, such
 *  that in conflict analysis, this change is treated like a branching decision
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPchgVarUb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   );

/** changes lower bound of variable in the given node; if possible, adjust bound to integral value; doesn't store any
 *  inference information in the bound change, such that in conflict analysis, this change is treated like a branching
 *  decision
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can only be called if @p scip is in stage \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgVarLbNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to change bound at, or NULL for current node */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   );

/** changes upper bound of variable in the given node; if possible, adjust bound to integral value; doesn't store any
 *  inference information in the bound change, such that in conflict analysis, this change is treated like a branching
 *  decision
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can only be called if @p scip is in stage \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgVarUbNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to change bound at, or NULL for current node */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   );

/** changes global lower bound of variable; if possible, adjust bound to integral value; also tightens the local bound,
 *  if the global bound is better than the local bound
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPchgVarLbGlobal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   );

/** changes global upper bound of variable; if possible, adjust bound to integral value; also tightens the local bound,
 *  if the global bound is better than the local bound
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPchgVarUbGlobal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   );

/** changes lazy lower bound of the variable, this is only possible if the variable is not in the LP yet
 *
 *  lazy bounds are bounds, that are enforced by constraints and the objective function; hence, these bounds do not need
 *  to be put into the LP explicitly.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note lazy bounds are useful for branch-and-price since the corresponding variable bounds are not part of the LP
 */
EXTERN
SCIP_RETCODE SCIPchgVarLbLazy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             lazylb              /**< the lazy lower bound to be set */
   );

/** changes lazy upper bound of the variable, this is only possible if the variable is not in the LP yet
 *
 *  lazy bounds are bounds, that are enforced by constraints and the objective function; hence, these bounds do not need
 *  to be put into the LP explicitly.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note lazy bounds are useful for branch-and-price since the corresponding variable bounds are not part of the LP
 */
EXTERN
SCIP_RETCODE SCIPchgVarUbLazy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             lazyub              /**< the lazy lower bound to be set */
   );

/** changes lower bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  doesn't store any inference information in the bound change, such that in conflict analysis, this change
 *  is treated like a branching decision
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPtightenVarLb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the new domain is empty */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   );

/** changes upper bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  doesn't store any inference information in the bound change, such that in conflict analysis, this change
 *  is treated like a branching decision
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPtightenVarUb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the new domain is empty */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   );

/** changes lower bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  the given inference constraint is stored, such that the conflict analysis is able to find out the reason
 *  for the deduction of the bound change
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPinferVarLbCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_CONS*            infercons,          /**< constraint that deduced the bound change, or NULL */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the bound change is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   );

/** changes upper bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  the given inference constraint is stored, such that the conflict analysis is able to find out the reason
 *  for the deduction of the bound change
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPinferVarUbCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_CONS*            infercons,          /**< constraint that deduced the bound change */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the bound change is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   );

/** depending on SCIP's stage, fixes binary variable in the problem, in preprocessing, or in current node;
 *  the given inference constraint is stored, such that the conflict analysis is able to find out the reason for the
 *  deduction of the fixing
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPinferBinvarCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< binary variable to fix */
   SCIP_Bool             fixedval,           /**< value to fix binary variable to */
   SCIP_CONS*            infercons,          /**< constraint that deduced the fixing */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the fixing is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the fixing tightened the local bounds, or NULL */
   );

/** changes lower bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  the given inference propagator is stored, such that the conflict analysis is able to find out the reason
 *  for the deduction of the bound change
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPinferVarLbProp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_PROP*            inferprop,          /**< propagator that deduced the bound change, or NULL */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the bound change is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   );

/** changes upper bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  the given inference propagator is stored, such that the conflict analysis is able to find out the reason
 *  for the deduction of the bound change
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPinferVarUbProp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_PROP*            inferprop,          /**< propagator that deduced the bound change */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the bound change is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   );

/** depending on SCIP's stage, fixes binary variable in the problem, in preprocessing, or in current node;
 *  the given inference propagator is stored, such that the conflict analysis is able to find out the reason for the
 *  deduction of the fixing
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPinferBinvarProp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< binary variable to fix */
   SCIP_Bool             fixedval,           /**< value to fix binary variable to */
   SCIP_PROP*            inferprop,          /**< propagator that deduced the fixing */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the fixing is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the fixing tightened the local bounds, or NULL */
   );

/** changes global lower bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current global bound; if possible, adjusts bound to integral value;
 *  also tightens the local bound, if the global bound is better than the local bound
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPtightenVarLbGlobal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the new domain is empty */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   );

/** changes global upper bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current global bound; if possible, adjusts bound to integral value;
 *  also tightens the local bound, if the global bound is better than the local bound
 *
 *  @warning If SCIP is in presolving stage, it can happen that the internal variable array (which can be accessed via
 *           SCIPgetVars()) gets resorted.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note During presolving, an integer variable whose bound changes to {0,1} is upgraded to a binary variable.
 */
EXTERN
SCIP_RETCODE SCIPtightenVarUbGlobal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the new domain is empty */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   );

/** for a multi-aggregated variable, returns the global lower bound computed by adding the global bounds from all aggregation variables
 * this global bound may be tighter than the one given by SCIPvarGetLbGlobal, since the latter is not updated if bounds of aggregation variables are changing
 * calling this function for a non-multi-aggregated variable results in a call to SCIPvarGetLbGlobal
 *
 * @return the global lower bound computed by adding the global bounds from all aggregation variables
 */
EXTERN
SCIP_Real SCIPcomputeVarLbGlobal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to compute the bound for */
   );

/** for a multi-aggregated variable, returns the global upper bound computed by adding the global bounds from all aggregation variables
 * this global bound may be tighter than the one given by SCIPvarGetUbGlobal, since the latter is not updated if bounds of aggregation variables are changing
 * calling this function for a non-multi-aggregated variable results in a call to SCIPvarGetUbGlobal
 *
 * @return the global upper bound computed by adding the global bounds from all aggregation variables
 */
EXTERN
SCIP_Real SCIPcomputeVarUbGlobal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to compute the bound for */
   );

/** for a multi-aggregated variable, returns the local lower bound computed by adding the local bounds from all aggregation variables
 * this local bound may be tighter than the one given by SCIPvarGetLbLocal, since the latter is not updated if bounds of aggregation variables are changing
 * calling this function for a non-multi-aggregated variable results in a call to SCIPvarGetLbLocal
 *
 * @return the local lower bound computed by adding the global bounds from all aggregation variables
 */
EXTERN
SCIP_Real SCIPcomputeVarLbLocal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to compute the bound for */
   );

/** for a multi-aggregated variable, returns the local upper bound computed by adding the local bounds from all aggregation variables
 * this local bound may be tighter than the one given by SCIPvarGetUbLocal, since the latter is not updated if bounds of aggregation variables are changing
 * calling this function for a non-multi-aggregated variable results in a call to SCIPvarGetUbLocal
 *
 * @return the local upper bound computed by adding the global bounds from all aggregation variables
 */
EXTERN
SCIP_Real SCIPcomputeVarUbLocal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to compute the bound for */
   );

#ifdef NDEBUG

/* In optimized mode, the function calls are overwritten by defines to reduce the number of function calls and
 * speed up the algorithms.
 */

#define SCIPcomputeVarLbGlobal(scip, var)  (SCIPvarGetStatus(var) == SCIP_VARSTATUS_MULTAGGR ? SCIPvarGetMultaggrLbGlobal(var, (scip)->set) : SCIPvarGetLbGlobal(var))
#define SCIPcomputeVarUbGlobal(scip, var)  (SCIPvarGetStatus(var) == SCIP_VARSTATUS_MULTAGGR ? SCIPvarGetMultaggrUbGlobal(var, (scip)->set) : SCIPvarGetUbGlobal(var))
#define SCIPcomputeVarLbLocal(scip, var)   (SCIPvarGetStatus(var) == SCIP_VARSTATUS_MULTAGGR ? SCIPvarGetMultaggrLbLocal(var, (scip)->set)  : SCIPvarGetLbLocal(var))
#define SCIPcomputeVarUbLocal(scip, var)   (SCIPvarGetStatus(var) == SCIP_VARSTATUS_MULTAGGR ? SCIPvarGetMultaggrUbLocal(var, (scip)->set)  : SCIPvarGetUbLocal(var))

#endif

/** returns solution value and index of variable lower bound that is closest to the variable's value in the given primal
 *  solution or current LP solution if no primal solution is given; returns an index of -1 if no variable lower bound is
 *  available
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can only be called if @p scip is in stage \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetVarClosestVlb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_SOL*             sol,                /**< primal solution, or NULL for LP solution */
   SCIP_Real*            closestvlb,         /**< pointer to store the value of the closest variable lower bound */
   int*                  closestvlbidx       /**< pointer to store the index of the closest variable lower bound */
   );

/** returns solution value and index of variable upper bound that is closest to the variable's value in the given primal solution;
 *  or current LP solution if no primal solution is given; returns an index of -1 if no variable upper bound is available
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can only be called if @p scip is in stage \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetVarClosestVub(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_SOL*             sol,                /**< primal solution, or NULL for LP solution */
   SCIP_Real*            closestvub,         /**< pointer to store the value of the closest variable lower bound */
   int*                  closestvubidx       /**< pointer to store the index of the closest variable lower bound */
   );

/** informs variable x about a globally valid variable lower bound x >= b*z + d with integer variable z;
 *  if z is binary, the corresponding valid implication for z is also added;
 *  if z is non-continuous and 1/b not too small, the corresponding valid upper/lower bound
 *  z <= (x-d)/b or z >= (x-d)/b (depending on the sign of of b) is added, too;
 *  improves the global bounds of the variable and the vlb variable if possible
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddVarVlb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_VAR*             vlbvar,             /**< variable z    in x >= b*z + d */
   SCIP_Real             vlbcoef,            /**< coefficient b in x >= b*z + d */
   SCIP_Real             vlbconstant,        /**< constant d    in x >= b*z + d */
   SCIP_Bool*            infeasible,         /**< pointer to store whether an infeasibility was detected */
   int*                  nbdchgs             /**< pointer to store the number of performed bound changes, or NULL */
   );


/** informs variable x about a globally valid variable upper bound x <= b*z + d with integer variable z;
 *  if z is binary, the corresponding valid implication for z is also added;
 *  if z is non-continuous and 1/b not too small, the corresponding valid lower/upper bound
 *  z >= (x-d)/b or z <= (x-d)/b (depending on the sign of of b) is added, too;
 *  improves the global bounds of the variable and the vlb variable if possible
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddVarVub(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_VAR*             vubvar,             /**< variable z    in x <= b*z + d */
   SCIP_Real             vubcoef,            /**< coefficient b in x <= b*z + d */
   SCIP_Real             vubconstant,        /**< constant d    in x <= b*z + d */
   SCIP_Bool*            infeasible,         /**< pointer to store whether an infeasibility was detected */
   int*                  nbdchgs             /**< pointer to store the number of performed bound changes, or NULL */
   );

/** informs binary variable x about a globally valid implication:  x == 0 or x == 1  ==>  y <= b  or  y >= b;
 *  also adds the corresponding implication or variable bound to the implied variable;
 *  if the implication is conflicting, the variable is fixed to the opposite value;
 *  if the variable is already fixed to the given value, the implication is performed immediately;
 *  if the implication is redundant with respect to the variables' global bounds, it is ignored
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddVarImplication(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Bool             varfixing,          /**< FALSE if y should be added in implications for x == 0, TRUE for x == 1 */
   SCIP_VAR*             implvar,            /**< variable y in implication y <= b or y >= b */
   SCIP_BOUNDTYPE        impltype,           /**< type       of implication y <= b (SCIP_BOUNDTYPE_UPPER)
                                              *                          or y >= b (SCIP_BOUNDTYPE_LOWER) */
   SCIP_Real             implbound,          /**< bound b    in implication y <= b or y >= b */
   SCIP_Bool*            infeasible,         /**< pointer to store whether an infeasibility was detected */
   int*                  nbdchgs             /**< pointer to store the number of performed bound changes, or NULL */
   );

/** adds a clique information to SCIP, stating that at most one of the given binary variables can be set to 1;
 *  if a variable appears twice in the same clique, the corresponding implications are performed
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddClique(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            vars,               /**< binary variables in the clique from which at most one can be set to 1 */
   SCIP_Bool*            values,             /**< values of the variables in the clique; NULL to use TRUE for all vars */
   int                   nvars,              /**< number of variables in the clique */
   SCIP_Bool*            infeasible,         /**< pointer to store whether an infeasibility was detected */
   int*                  nbdchgs             /**< pointer to store the number of performed bound changes, or NULL */
   );

/** calculates a partition of the given set of binary variables into cliques;
 *  afterwards the output array contains one value for each variable, such that two variables got the same value iff they
 *  were assigned to the same clique;
 *  the first variable is always assigned to clique 0, and a variable can only be assigned to clique i if at least one of
 *  the preceding variables was assigned to clique i-1;
 *  for each clique at most 1 variables can be set to TRUE in a feasible solution;
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcalcCliquePartition(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_VAR**const       vars,               /**< binary variables in the clique from which at most one can be set to 1 */
   int const             nvars,              /**< number of variables in the clique */
   int*const             cliquepartition,    /**< array of length nvars to store the clique partition */
   int*const             ncliques            /**< pointer to store the number of cliques actually contained in the partition */
   );

/** calculates a partition of the given set of binary variables into negated cliques;
 *  afterwards the output array contains one value for each variable, such that two variables got the same value iff they
 *  were assigned to the same negated clique;
 *  the first variable is always assigned to clique 0 and a variable can only be assigned to clique i if at least one of
 *  the preceding variables was assigned to clique i-1;
 *  for each clique with n_c variables at least n_c-1 variables can be set to TRUE in a feasible solution;
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcalcNegatedCliquePartition(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_VAR**const       vars,               /**< binary variables in the clique from which at most one can be set to 1 */
   int const             nvars,              /**< number of variables in the clique */
   int*const             cliquepartition,    /**< array of length nvars to store the clique partition */
   int*const             ncliques            /**< pointer to store the number of cliques actually contained in the partition */
   );

/** gets the number of cliques in the clique table
 *
 *  @return number of cliques in the clique table
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetNCliques(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the array of cliques in the clique table
 *
 *  @return array of cliques in the clique table
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_CLIQUE** SCIPgetCliques(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether there is a clique that contains both given variable/value pairs;
 *  the variables must be active binary variables;
 *  if regardimplics is FALSE, only the cliques in the clique table are looked at;
 *  if regardimplics is TRUE, both the cliques and the implications of the implication graph are regarded
 *
 *  @return TRUE, if there is a clique that contains both variable/clique pairs; FALSE, otherwise
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @note a variable with it's negated variable are NOT! in a clique
 *  @note a variable with itself are in a clique
 */
EXTERN
SCIP_Bool SCIPhaveVarsCommonClique(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var1,               /**< first variable */
   SCIP_Bool             value1,             /**< value of first variable */
   SCIP_VAR*             var2,               /**< second variable */
   SCIP_Bool             value2,             /**< value of second variable */
   SCIP_Bool             regardimplics       /**< should the implication graph also be searched for a clique? */
   );

/** writes the clique graph to a gml file
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @note there can be duplicated arcs in the output file
 */
EXTERN
SCIP_RETCODE SCIPwriteCliqueGraph(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           fname,              /**< name of file */
   SCIP_Bool             writeimplications   /**< should we write the binary implications */
   );


/** sets the branch factor of the variable; this value can be used in the branching methods to scale the score
 *  values of the variables; higher factor leads to a higher probability that this variable is chosen for branching
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgVarBranchFactor(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             branchfactor        /**< factor to weigh variable's branching score with */
   );

/** scales the branch factor of the variable with the given value
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPscaleVarBranchFactor(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             scale               /**< factor to scale variable's branching factor with */
   );

/** adds the given value to the branch factor of the variable
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddVarBranchFactor(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             addfactor           /**< value to add to the branch factor of the variable */
   );

/** sets the branch priority of the variable; variables with higher branch priority are always preferred to variables
 *  with lower priority in selection of branching variable
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *
 * @note the default branching priority is 0
 */
EXTERN
SCIP_RETCODE SCIPchgVarBranchPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   int                   branchpriority      /**< branch priority of the variable */
   );

/** changes the branch priority of the variable to the given value, if it is larger than the current priority
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPupdateVarBranchPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   int                   branchpriority      /**< new branch priority of the variable, if it is larger than current priority */
   );

/** adds the given value to the branch priority of the variable
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddVarBranchPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   int                   addpriority         /**< value to add to the branch priority of the variable */
   );

/** sets the branch direction of the variable (-1: prefer downwards branch, 0: automatic selection, +1: prefer upwards
 *  branch)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgVarBranchDirection(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        branchdirection     /**< preferred branch direction of the variable (downwards, upwards, auto) */
   );

/** changes type of variable in the problem;
 *
 *  @warning This type change might change the variable array returned from SCIPgetVars() and SCIPgetVarsData();
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *
 *  @note If SCIP is already beyond the SCIP_STAGE_PROBLEM and a original variable is passed, the variable type of the
 *        corresponding transformed variable is changed; the type of the original variable does not change
 *
 *  @note If the type changes from a continuous variable to a non-continuous variable the bounds of the variable get
 *        adjusted w.r.t. to integrality information
 */
EXTERN
SCIP_RETCODE SCIPchgVarType(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_VARTYPE          vartype,            /**< new type of variable */
   SCIP_Bool*            infeasible          /**< pointer to store whether an infeasibility was detected (, due to
                                              *   integrality condition of the new variable type) */
   );

/** in problem creation and solving stage, both bounds of the variable are set to the given value;
 *  in presolving stage, the variable is converted into a fixed variable, and bounds are changed respectively;
 *  conversion into a fixed variable changes the vars array returned from SCIPgetVars() and SCIPgetVarsData(),
 *  and also renders arrays returned from the SCIPvarGetImpl...() methods invalid
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPfixVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to fix */
   SCIP_Real             fixedval,           /**< value to fix variable to */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the fixing is infeasible */
   SCIP_Bool*            fixed               /**< pointer to store whether the fixing was performed (variable was unfixed) */
   );

/** From a given equality a*x + b*y == c, aggregates one of the variables and removes it from the set of
 *  active problem variables. This changes the vars array returned from SCIPgetVars() and SCIPgetVarsData(),
 *  and also renders the arrays returned from the SCIPvarGetImpl...() methods for the two variables invalid.
 *  In the first step, the equality is transformed into an equality with active problem variables
 *  a'*x' + b'*y' == c'. If x' == y', this leads to the detection of redundancy if a' == -b' and c' == 0,
 *  of infeasibility, if a' == -b' and c' != 0, or to a variable fixing x' == c'/(a'+b') (and possible
 *  infeasibility) otherwise.
 *  In the second step, the variable to be aggregated is chosen among x' and y', prefering a less strict variable
 *  type as aggregation variable (i.e. continuous variables are preferred over implicit integers, implicit integers
 *  over integers, and integers over binaries). If none of the variables is continuous, it is tried to find an integer
 *  aggregation (i.e. integral coefficients a'' and b'', such that a''*x' + b''*y' == c''). This can lead to
 *  the detection of infeasibility (e.g. if c'' is fractional), or to a rejection of the aggregation (denoted by
 *  aggregated == FALSE), if the resulting integer coefficients are too large and thus numerically instable.
 *
 *  The output flags have the following meaning:
 *  - infeasible: the problem is infeasible
 *  - redundant:  the equality can be deleted from the constraint set
 *  - aggregated: the aggregation was successfully performed (the variables were not aggregated before)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can only be called if @p scip is in stage \ref SCIP_STAGE_PRESOLVING
 */
EXTERN
SCIP_RETCODE SCIPaggregateVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             varx,               /**< variable x in equality a*x + b*y == c */
   SCIP_VAR*             vary,               /**< variable y in equality a*x + b*y == c */
   SCIP_Real             scalarx,            /**< multiplier a in equality a*x + b*y == c */
   SCIP_Real             scalary,            /**< multiplier b in equality a*x + b*y == c */
   SCIP_Real             rhs,                /**< right hand side c in equality a*x + b*y == c */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the aggregation is infeasible */
   SCIP_Bool*            redundant,          /**< pointer to store whether the equality is (now) redundant */
   SCIP_Bool*            aggregated          /**< pointer to store whether the aggregation was successful */
   );

/** converts variable into multi-aggregated variable; this changes the variable array returned from
 *  SCIPgetVars() and SCIPgetVarsData();
 *
 *  @warning The integrality condition is not checked anymore on the multi-aggregated variable. You must not
 *           multi-aggregate an integer variable without being sure, that integrality on the aggregation variables
 *           implies integrality on the aggregated variable.
 *
 *  The output flags have the following meaning:
 *  - infeasible: the problem is infeasible
 *  - aggregated: the aggregation was successfully performed (the variables were not aggregated before)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can only be called if @p scip is in stage \ref SCIP_STAGE_PRESOLVING
 */
EXTERN
SCIP_RETCODE SCIPmultiaggregateVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable x to aggregate */
   int                   naggvars,           /**< number n of variables in aggregation x = a_1*y_1 + ... + a_n*y_n + c */
   SCIP_VAR**            aggvars,            /**< variables y_i in aggregation x = a_1*y_1 + ... + a_n*y_n + c */
   SCIP_Real*            scalars,            /**< multipliers a_i in aggregation x = a_1*y_1 + ... + a_n*y_n + c */
   SCIP_Real             constant,           /**< constant shift c in aggregation x = a_1*y_1 + ... + a_n*y_n + c */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the aggregation is infeasible */
   SCIP_Bool*            aggregated          /**< pointer to store whether the aggregation was successful */
   );

/** returns whether aggregation of variables is not allowed */
EXTERN
SCIP_Bool SCIPdoNotAggr(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether multi-aggregation is disabled */
EXTERN
SCIP_Bool SCIPdoNotMultaggr(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether variable is not allowed to be multi-aggregated */
EXTERN
SCIP_Bool SCIPdoNotMultaggrVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable x to aggregate */
   );

/** marks the variable that it must not be multi-aggregated
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 */
EXTERN
SCIP_RETCODE SCIPmarkDoNotMultaggrVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to delete */
   );

/** enables the collection of statistics for a variable
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
void SCIPenableVarHistory(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** disables the collection of any statistic for a variable
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
void SCIPdisableVarHistory(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** updates the pseudo costs of the given variable and the global pseudo costs after a change of "solvaldelta" in the
 *  variable's solution value and resulting change of "objdelta" in the in the LP's objective value;
 *  the update is ignored, if the objective value difference is infinite
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPupdateVarPseudocost(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             solvaldelta,        /**< difference of variable's new LP value - old LP value */
   SCIP_Real             objdelta,           /**< difference of new LP's objective value - old LP's objective value */
   SCIP_Real             weight              /**< weight in (0,1] of this update in pseudo cost sum */
   );

/** gets the variable's pseudo cost value for the given change of the variable's LP value
 *
 *  @return the variable's pseudo cost value for the given change of the variable's LP value
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarPseudocostVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             solvaldelta         /**< difference of variable's new LP value - old LP value */
   );

/** gets the variable's pseudo cost value for the given change of the variable's LP value,
 *  only using the pseudo cost information of the current run
 *
 *  @return the variable's pseudo cost value for the given change of the variable's LP value,
 *  only using the pseudo cost information of the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarPseudocostValCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             solvaldelta         /**< difference of variable's new LP value - old LP value */
   );

/** gets the variable's pseudo cost value for the given direction
 *
 *  @return the variable's pseudo cost value for the given direction
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarPseudocost(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** gets the variable's pseudo cost value for the given direction,
 *  only using the pseudo cost information of the current run
 *
 *  @return the variable's pseudo cost value for the given direction,
 *  only using the pseudo cost information of the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarPseudocostCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** gets the variable's (possible fractional) number of pseudo cost updates for the given direction
 *
 *  @return the variable's (possible fractional) number of pseudo cost updates for the given direction
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarPseudocostCount(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** gets the variable's (possible fractional) number of pseudo cost updates for the given direction,
 *  only using the pseudo cost information of the current run
 *
 *  @return the variable's (possible fractional) number of pseudo cost updates for the given direction,
 *  only using the pseudo cost information of the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarPseudocostCountCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** gets the variable's pseudo cost score value for the given LP solution value
 *
 *  @return the variable's pseudo cost score value for the given LP solution value
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarPseudocostScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             solval              /**< variable's LP solution value */
   );

/** gets the variable's pseudo cost score value for the given LP solution value,
 *  only using the pseudo cost information of the current run
 *
 *  @return the variable's pseudo cost score value for the given LP solution value,
 *  only using the pseudo cost information of the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarPseudocostScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             solval              /**< variable's LP solution value */
   );

/** returns the variable's VSIDS value
 *
 *  @return the variable's VSIDS value
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarVSIDS(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the variable's VSIDS value only using conflicts of the current run
 *
 *  @return the variable's VSIDS value only using conflicts of the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarVSIDSCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the variable's conflict score value
 *
 *  @return the variable's conflict score value
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarConflictScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the variable's conflict score value only using conflicts of the current run
 *
 *  @return the variable's conflict score value only using conflicts of the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarConflictScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the variable's conflict length score
 *
 *  @return the variable's conflict length score
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarConflictlengthScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the variable's conflict length score only using conflicts of the current run
 *
 *  @return the variable's conflict length score only using conflicts of the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarConflictlengthScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the variable's average conflict length
 *
 *  @return the variable's average conflict length
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgConflictlength(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the variable's average  conflict length only using conflicts of the current run
 *
 *  @return the variable's average conflict length only using conflicts of the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgConflictlengthCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the average number of inferences found after branching on the variable in given direction;
 *  if branching on the variable in the given direction was yet evaluated, the average number of inferences
 *  over all variables for branching in the given direction is returned
 *
 *  @return the average number of inferences found after branching on the variable in given direction
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgInferences(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the average number of inferences found after branching on the variable in given direction in the current run;
 *  if branching on the variable in the given direction was yet evaluated, the average number of inferences
 *  over all variables for branching in the given direction is returned
 *
 *  @return the average number of inferences found after branching on the variable in given direction in the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgInferencesCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the variable's average inference score value
 *
 *  @return the variable's average inference score value
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgInferenceScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the variable's average inference score value only using inferences of the current run
 *
 *  @return the variable's average inference score value only using inferences of the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgInferenceScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** initializes the upwards and downwards pseudocosts, conflict scores, conflict lengths, inference scores, cutoff scores
 *  of a variable to the given values 
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPinitVarBranchStats(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable which should be initialized */
   SCIP_Real             downpscost,         /**< value to which pseudocosts for downwards branching should be initialized */
   SCIP_Real             uppscost,           /**< value to which pseudocosts for upwards branching should be initialized */
   SCIP_Real             downvsids,          /**< value to which VSIDS score for downwards branching should be initialized */
   SCIP_Real             upvsids,            /**< value to which VSIDS score for upwards branching should be initialized */
   SCIP_Real             downconflen,        /**< value to which conflict length score for downwards branching should be initialized */
   SCIP_Real             upconflen,          /**< value to which conflict length score for upwards branching should be initialized */
   SCIP_Real             downinfer,          /**< value to which inference counter for downwards branching should be initialized */
   SCIP_Real             upinfer,            /**< value to which inference counter for upwards branching should be initialized */
   SCIP_Real             downcutoff,         /**< value to which cutoff counter for downwards branching should be initialized */
   SCIP_Real             upcutoff            /**< value to which cutoff counter for upwards branching should be initialized */
   );

/** returns the average number of cutoffs found after branching on the variable in given direction;
 *  if branching on the variable in the given direction was yet evaluated, the average number of cutoffs
 *  over all variables for branching in the given direction is returned
 *
 *  @return the average number of cutoffs found after branching on the variable in given direction
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgCutoffs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the average number of cutoffs found after branching on the variable in given direction in the current run;
 *  if branching on the variable in the given direction was yet evaluated, the average number of cutoffs
 *  over all variables for branching in the given direction is returned
 *
 *  @return the average number of cutoffs found after branching on the variable in given direction in the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgCutoffsCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the variable's average cutoff score value
 *
 *  @return the variable's average cutoff score value
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgCutoffScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the variable's average cutoff score value, only using cutoffs of the current run
 *
 *  @return the variable's average cutoff score value, only using cutoffs of the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgCutoffScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the variable's average inference/cutoff score value, weighting the cutoffs of the variable with the given
 *  factor
 *
 *  @return the variable's average inference/cutoff score value
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgInferenceCutoffScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             cutoffweight        /**< factor to weigh average number of cutoffs in branching score */
   );

/** returns the variable's average inference/cutoff score value, weighting the cutoffs of the variable with the given
 *  factor, only using inferences and cutoffs of the current run
 *
 *  @return the variable's average inference/cutoff score value, only using inferences and cutoffs of the current run
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetVarAvgInferenceCutoffScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             cutoffweight        /**< factor to weigh average number of cutoffs in branching score */
   );

/** outputs variable information to file stream via the message system
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note If the message handler is set to a NULL pointer nothing will be printed
 */
EXTERN
SCIP_RETCODE SCIPprintVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/**@} */




/*
 * conflict analysis methods
 */

/**@name Conflict Analysis Methods */
/**@{ */

/** return TRUE if conflict analysis is applicable; In case the function return FALSE there is no need to initialize the
 *  conflict analysis since it will not be applied
 *
 *  @return return TRUE if conflict analysis is applicable; In case the function return FALSE there is no need to initialize the
 *          conflict analysis since it will not be applied
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_Bool SCIPisConflictAnalysisApplicable(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** initializes the conflict analysis by clearing the conflict candidate queue; this method must be called before you
 *  enter the conflict variables by calling SCIPaddConflictLb(), SCIPaddConflictUb(), SCIPaddConflictBd(),
 *  SCIPaddConflictRelaxedLb(), SCIPaddConflictRelaxedUb(), SCIPaddConflictRelaxedBd(), or SCIPaddConflictBinvar();
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPinitConflictAnalysis(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** adds lower bound of variable at the time of the given bound change index to the conflict analysis' candidate storage;
 *  this method should be called in one of the following two cases:
 *   1. Before calling the SCIPanalyzeConflict() method, SCIPaddConflictLb() should be called for each lower bound
 *      that led to the conflict (e.g. the infeasibility of globally or locally valid constraint).
 *   2. In the propagation conflict resolving method of a constraint handler, SCIPaddConflictLb() should be called
 *      for each lower bound, whose current assignment led to the deduction of the given conflict bound.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPaddConflictLb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable whose lower bound should be added to conflict candidate queue */
   SCIP_BDCHGIDX*        bdchgidx            /**< bound change index representing time on path to current node, when the
                                              *   conflicting bound was valid, NULL for current local bound */
   );

/** adds lower bound of variable at the time of the given bound change index to the conflict analysis' candidate storage
 *  with the additional information of a relaxed lower bound; this relaxed lower bound is the one which would be enough
 *  to explain a certain bound change;
 *  this method should be called in one of the following two cases:
 *   1. Before calling the SCIPanalyzeConflict() method, SCIPaddConflictRelaxedLb() should be called for each (relaxed) lower bound
 *      that led to the conflict (e.g. the infeasibility of globally or locally valid constraint).
 *   2. In the propagation conflict resolving method of a constraint handler, SCIPaddConflictRelexedLb() should be called
 *      for each (relaxed) lower bound, whose current assignment led to the deduction of the given conflict bound.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPaddConflictRelaxedLb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable whose lower bound should be added to conflict candidate queue */
   SCIP_BDCHGIDX*        bdchgidx,           /**< bound change index representing time on path to current node, when the
                                              *   conflicting bound was valid, NULL for current local bound */
   SCIP_Real             relaxedlb           /**< the relaxed lower bound */
   );

/** adds upper bound of variable at the time of the given bound change index to the conflict analysis' candidate storage;
 *  this method should be called in one of the following two cases:
 *   1. Before calling the SCIPanalyzeConflict() method, SCIPaddConflictUb() should be called for each upper bound that
 *      led to the conflict (e.g. the infeasibility of globally or locally valid constraint).
 *   2. In the propagation conflict resolving method of a constraint handler, SCIPaddConflictUb() should be called for
 *      each upper bound, whose current assignment led to the deduction of the given conflict bound.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPaddConflictUb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable whose upper bound should be added to conflict candidate queue */
   SCIP_BDCHGIDX*        bdchgidx            /**< bound change index representing time on path to current node, when the
                                              *   conflicting bound was valid, NULL for current local bound */
   );

/** adds upper bound of variable at the time of the given bound change index to the conflict analysis' candidate storage
 *  with the additional information of a relaxed upper bound; this relaxed upper bound is the one which would be enough
 *  to explain a certain bound change;
 *  this method should be called in one of the following two cases:
 *   1. Before calling the SCIPanalyzeConflict() method, SCIPaddConflictRelaxedUb() should be called for each (relaxed) upper
 *      bound that led to the conflict (e.g. the infeasibility of globally or locally valid constraint).
 *   2. In the propagation conflict resolving method of a constraint handler, SCIPaddConflictRelaxedUb() should be
 *      called for each (relaxed) upper bound, whose current assignment led to the deduction of the given conflict
 *      bound.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPaddConflictRelaxedUb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable whose upper bound should be added to conflict candidate queue */
   SCIP_BDCHGIDX*        bdchgidx,           /**< bound change index representing time on path to current node, when the
                                              *   conflicting bound was valid, NULL for current local bound */
   SCIP_Real             relaxedub           /**< the relaxed upper bound */
   );

/** adds lower or upper bound of variable at the time of the given bound change index to the conflict analysis' candidate
 *  storage; this method should be called in one of the following two cases:
 *   1. Before calling the SCIPanalyzeConflict() method, SCIPaddConflictBd() should be called for each bound
 *      that led to the conflict (e.g. the infeasibility of globally or locally valid constraint).
 *   2. In the propagation conflict resolving method of a constraint handler, SCIPaddConflictBd() should be called
 *      for each bound, whose current assignment led to the deduction of the given conflict bound.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPaddConflictBd(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable whose upper bound should be added to conflict candidate queue */
   SCIP_BOUNDTYPE        boundtype,          /**< the type of the conflicting bound (lower or upper bound) */
   SCIP_BDCHGIDX*        bdchgidx            /**< bound change index representing time on path to current node, when the
                                              *   conflicting bound was valid, NULL for current local bound */
   );

/** adds lower or upper bound of variable at the time of the given bound change index to the conflict analysis'
 *  candidate storage; with the additional information of a relaxed upper bound; this relaxed upper bound is the one
 *  which would be enough to explain a certain bound change;
 *  this method should be called in one of the following two cases:
 *   1. Before calling the SCIPanalyzeConflict() method, SCIPaddConflictRelaxedBd() should be called for each (relaxed)
 *      bound that led to the conflict (e.g. the infeasibility of globally or locally valid constraint).
 *   2. In the propagation conflict resolving method of a constraint handler, SCIPaddConflictRelaxedBd() should be
 *      called for each (relaxed) bound, whose current assignment led to the deduction of the given conflict bound.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPaddConflictRelaxedBd(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable whose upper bound should be added to conflict candidate queue */
   SCIP_BOUNDTYPE        boundtype,          /**< the type of the conflicting bound (lower or upper bound) */
   SCIP_BDCHGIDX*        bdchgidx,           /**< bound change index representing time on path to current node, when the
                                              *   conflicting bound was valid, NULL for current local bound */
   SCIP_Real             relaxedbd           /**< the relaxed bound */
   );

/** adds changed bound of fixed binary variable to the conflict analysis' candidate storage;
 *  this method should be called in one of the following two cases:
 *   1. Before calling the SCIPanalyzeConflict() method, SCIPaddConflictBinvar() should be called for each fixed binary
 *      variable that led to the conflict (e.g. the infeasibility of globally or locally valid constraint).
 *   2. In the propagation conflict resolving method of a constraint handler, SCIPaddConflictBinvar() should be called
 *      for each binary variable, whose current fixing led to the deduction of the given conflict bound.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPaddConflictBinvar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< binary variable whose changed bound should be added to conflict queue */
   );

/** checks if the given variable is already part of the current conflict set or queued for resolving with the same or
 *  even stronger bound
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPisConflictVarUsed(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable whose upper bound should be added to conflict candidate queue */
   SCIP_BOUNDTYPE        boundtype,          /**< the type of the conflicting bound (lower or upper bound) */
   SCIP_BDCHGIDX*        bdchgidx,           /**< bound change index representing time on path to current node, when the
                                              *   conflicting bound was valid, NULL for current local bound */
   SCIP_Bool*            used                /**< pointer to store if the variable is already used */
   );

/** returns the conflict lower bound if the variable is present in the current conflict set; otherwise the global lower
 *  bound
 *
 *  @return returns the conflict lower bound if the variable is present in the current conflict set; otherwise the global lower
 *          bound
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_Real SCIPgetConflictVarLb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the conflict upper bound if the variable is present in the current conflict set; otherwise minus global
 *  upper bound
 *
 *  @return returns the conflict upper bound if the variable is present in the current conflict set; otherwise minus global
 *          upper bound
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_Real SCIPgetConflictVarUb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** analyzes conflict bounds that were added after a call to SCIPinitConflictAnalysis() with calls to
 *  SCIPaddConflictLb(), SCIPaddConflictUb(), SCIPaddConflictBd(), SCIPaddConflictRelaxedLb(),
 *  SCIPaddConflictRelaxedUb(), SCIPaddConflictRelaxedBd(), or SCIPaddConflictBinvar(); on success, calls the conflict
 *  handlers to create a conflict constraint out of the resulting conflict set; the given valid depth must be a depth
 *  level, at which the conflict set defined by calls to SCIPaddConflictLb(), SCIPaddConflictUb(), SCIPaddConflictBd(),
 *  SCIPaddConflictRelaxedLb(), SCIPaddConflictRelaxedUb(), SCIPaddConflictRelaxedBd(), and SCIPaddConflictBinvar() is
 *  valid for the whole subtree; if the conflict was found by a violated constraint, use SCIPanalyzeConflictCons()
 *  instead of SCIPanalyzeConflict() to make sure, that the correct valid depth is used
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPanalyzeConflict(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   validdepth,         /**< minimal depth level at which the initial conflict set is valid */
   SCIP_Bool*            success             /**< pointer to store whether a conflict constraint was created, or NULL */
   );

/** analyzes conflict bounds that were added with calls to SCIPaddConflictLb(), SCIPaddConflictUb(),
 *  SCIPaddConflictBd(), SCIPaddConflictRelaxedLb(), SCIPaddConflictRelaxedUb(), SCIPaddConflictRelaxedBd(), or
 *  SCIPaddConflictBinvar(); on success, calls the conflict handlers to create a conflict constraint out of the
 *  resulting conflict set; the given constraint must be the constraint that detected the conflict, i.e. the constraint
 *  that is infeasible in the local bounds of the initial conflict set (defined by calls to SCIPaddConflictLb(),
 *  SCIPaddConflictUb(), SCIPaddConflictBd(), SCIPaddConflictRelaxedLb(), SCIPaddConflictRelaxedUb(),
 *  SCIPaddConflictRelaxedBd(), and SCIPaddConflictBinvar())
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note SCIP stage does not get changed
 */
EXTERN
SCIP_RETCODE SCIPanalyzeConflictCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint that detected the conflict */
   SCIP_Bool*            success             /**< pointer to store whether a conflict constraint was created, or NULL */
   );

/**@} */




/*
 * constraint methods
 */

/**@name Constraint Methods */
/**@{ */

/** creates and captures a constraint of the given constraint handler
 *
 *  @warning If a constraint is marked to be checked for feasibility but not to be enforced, a LP or pseudo solution may
 *           be declared feasible even if it violates this particular constraint.  This constellation should only be
 *           used, if no LP or pseudo solution can violate the constraint -- e.g. if a local constraint is redundant due
 *           to the variable's local bounds.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @note the constraint gets captured, hence at one point you have to release it using the method SCIPreleaseCons()
 */
EXTERN
SCIP_RETCODE SCIPcreateCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to constraint */
   const char*           name,               /**< name of constraint */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler for this constraint */
   SCIP_CONSDATA*        consdata,           /**< data for this specific constraint */
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
   );

/** parses constraint information (in cip format) out of a string; if the parsing process was successful a constraint is
 *  creates and captures;
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @warning If a constraint is marked to be checked for feasibility but not to be enforced, a LP or pseudo solution may
 *           be declared feasible even if it violates this particular constraint.  This constellation should only be
 *           used, if no LP or pseudo solution can violate the constraint -- e.g. if a local constraint is redundant due
 *           to the variable's local bounds.
 */
EXTERN
SCIP_RETCODE SCIPparseCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to store constraint */
   const char*           str,                /**< string to parse for constraint */
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
   SCIP_Bool             stickingatnode,     /**< should the constraint always be kept at the node where it was added, even
                                              *   if it may be moved to a more global node?
                                              *   Usually set to FALSE. Set to TRUE to for constraints that represent node data. */
   SCIP_Bool*            success             /**< pointer to store if the paring process was successful */
   );

/** increases usage counter of constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPcaptureCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint to capture */
   );

/** decreases usage counter of constraint, if the usage pointer reaches zero the constraint gets freed
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note the pointer of the constraint will be NULLed
 */
EXTERN
SCIP_RETCODE SCIPreleaseCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons                /**< pointer to constraint */
   );

/** change constraint name
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 */
EXTERN
SCIP_RETCODE SCIPchgConsName(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   const char*           name                /**< new name of constraint */
   );

/** sets the initial flag of the given constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetConsInitial(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             initial             /**< new value */
   );

/** sets the separate flag of the given constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetConsSeparated(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             separate            /**< new value */
   );

/** sets the enforce flag of the given constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetConsEnforced(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             enforce             /**< new value */
   );

/** sets the check flag of the given constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetConsChecked(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             check               /**< new value */
   );

/** sets the propagate flag of the given constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetConsPropagated(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             propagate           /**< new value */
   );

/** sets the local flag of the given constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetConsLocal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             local               /**< new value */
   );

/** sets the modifiable flag of the given constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPsetConsModifiable(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             modifiable          /**< new value */
   );

/** sets the dynamic flag of the given constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetConsDynamic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             dynamic             /**< new value */
   );

/** sets the removable flag of the given constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetConsRemovable(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             removable           /**< new value */
   );

/** sets the stickingatnode flag of the given constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetConsStickingAtNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             stickingatnode      /**< new value */
   );

/** updates the flags of the first constraint according to the ones of the second constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPupdateConsFlags(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons0,              /**< constraint that should stay */
   SCIP_CONS*            cons1               /**< constraint that should be deleted */
   );

/** gets and captures transformed constraint of a given constraint; if the constraint is not yet transformed,
 *  a new transformed constraint for this constraint is created
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPtransformCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to get/create transformed constraint for */
   SCIP_CONS**           transcons           /**< pointer to store the transformed constraint */
   );

/** gets and captures transformed constraints for an array of constraints;
 *  if a constraint in the array is not yet transformed, a new transformed constraint for this constraint is created;
 *  it is possible to call this method with conss == transconss
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPtransformConss(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nconss,             /**< number of constraints to get/create transformed constraints for */
   SCIP_CONS**           conss,              /**< array with constraints to get/create transformed constraints for */
   SCIP_CONS**           transconss          /**< array to store the transformed constraints */
   );

/** gets corresponding transformed constraint of a given constraint;
 *  returns NULL as transcons, if transformed constraint is not yet existing
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPgetTransformedCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to get the transformed constraint for */
   SCIP_CONS**           transcons           /**< pointer to store the transformed constraint */
   );

/** gets corresponding transformed constraints for an array of constraints;
 *  stores NULL in a transconss slot, if the transformed constraint is not yet existing;
 *  it is possible to call this method with conss == transconss, but remember that constraints that are not
 *  yet transformed will be replaced with NULL
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPgetTransformedConss(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nconss,             /**< number of constraints to get the transformed constraints for */
   SCIP_CONS**           conss,              /**< constraints to get the transformed constraints for */
   SCIP_CONS**           transconss          /**< array to store the transformed constraints */
   );

/** adds given value to age of constraint, but age can never become negative;
 *  should be called
 *   - in constraint separation, if no cut was found for this constraint,
 *   - in constraint enforcing, if constraint was feasible, and
 *   - in constraint propagation, if no domain reduction was deduced;
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPaddConsAge(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Real             deltaage            /**< value to add to the constraint's age */
   );

/** increases age of constraint by 1.0;
 *  should be called
 *   - in constraint separation, if no cut was found for this constraint,
 *   - in constraint enforcing, if constraint was feasible, and
 *   - in constraint propagation, if no domain reduction was deduced;
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPincConsAge(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   );

/** resets age of constraint to zero;
 *  should be called
 *   - in constraint separation, if a cut was found for this constraint,
 *   - in constraint enforcing, if the constraint was violated, and
 *   - in constraint propagation, if a domain reduction was deduced;
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPresetConsAge(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   );

/** enables constraint's separation, propagation, and enforcing capabilities
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPenableCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   );

/** disables constraint's separation, propagation, and enforcing capabilities, s.t. the constraint is not propagated,
 *  separated, and enforced anymore until it is enabled again with a call to SCIPenableCons();
 *  in contrast to SCIPdelConsLocal() and SCIPdelConsNode(), the disabling is not associated to a node in the tree and
 *  does not consume memory; therefore, the constraint is neither automatically enabled on leaving the node nor
 *  automatically disabled again on entering the node again;
 *  note that the constraints enforcing capabilities are necessary for the solution's feasibility, if the constraint
 *  is a model constraint; that means, you must be sure that the constraint cannot be violated in the current subtree,
 *  and you have to enable it again manually by calling SCIPenableCons(), if this subtree is left (e.g. by using
 *  an appropriate event handler that watches the corresponding variables' domain changes)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPdisableCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   );

/** enables constraint's separation capabilities
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPenableConsSeparation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   );

/** disables constraint's separation capabilities s.t. the constraint is not propagated anymore until the separation
 *  is enabled again with a call to SCIPenableConsSeparation(); in contrast to SCIPdelConsLocal() and SCIPdelConsNode(),
 *  the disabling is not associated to a node in the tree and does not consume memory; therefore, the constraint
 *  is neither automatically enabled on leaving the node nor automatically disabled again on entering the node again
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPdisableConsSeparation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   );

/** enables constraint's propagation capabilities
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPenableConsPropagation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   );

/** disables constraint's propagation capabilities s.t. the constraint is not propagated anymore until the propagation
 *  is enabled again with a call to SCIPenableConsPropagation(); in contrast to SCIPdelConsLocal() and SCIPdelConsNode(),
 *  the disabling is not associated to a node in the tree and does not consume memory; therefore, the constraint
 *  is neither automatically enabled on leaving the node nor automatically disabled again on entering the node again
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPdisableConsPropagation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   );


/** marks constraint to be propagated
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  @note if a constraint is marked to be propagated, the age of the constraint will be ignored for propagation
 */
EXTERN
SCIP_RETCODE SCIPmarkConsPropagate(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   );

/** unmarks the constraint to be propagated
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPunmarkConsPropagate(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   );

/** adds given values to lock status of the constraint and updates the rounding locks of the involved variables
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPaddConsLocks(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   int                   nlockspos,          /**< increase in number of rounding locks for constraint */
   int                   nlocksneg           /**< increase in number of rounding locks for constraint's negation */
   );

/** checks single constraint for feasibility of the given solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPcheckCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to check */
   SCIP_SOL*             sol,                /**< primal CIP solution */
   SCIP_Bool             checkintegrality,   /**< has integrality to be checked? */
   SCIP_Bool             checklprows,        /**< have current LP rows (both local and global) to be checked? */
   SCIP_Bool             printreason,        /**< should the reason for the violation be printed? */
   SCIP_RESULT*          result              /**< pointer to store the result of the callback method */
   );

/** enforces single constraint for a given pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note This is an advanced method and should be used with caution.  It may only be called for constraints that were not
 *        added to SCIP beforehand.
 */
EXTERN
SCIP_RETCODE SCIPenfopsCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to enforce */
   SCIP_Bool             solinfeasible,      /**< was the solution already declared infeasible by a constraint handler? */
   SCIP_Bool             objinfeasible,      /**< is the solution infeasible anyway due to violating lower objective bound? */
   SCIP_RESULT*          result              /**< pointer to store the result of the callback method */
   );

/** enforces single constraint for a given LP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note This is an advanced method and should be used with caution.  It may only be called for constraints that were not
 *        added to SCIP beforehand.
 */
EXTERN
SCIP_RETCODE SCIPenfolpCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to enforce */
   SCIP_Bool             solinfeasible,      /**< was the solution already declared infeasible by a constraint handler? */
   SCIP_RESULT*          result              /**< pointer to store the result of the callback method */
   );

/** calls LP initialization method for single constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note This is an advanced method and should be used with caution.  It may only be called for constraints that were not
 *        added to SCIP beforehand.
 */
EXTERN
SCIP_RETCODE SCIPinitlpCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint to initialize */
   );

/** calls separation method of single constraint for LP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note This is an advanced method and should be used with caution.
 */
EXTERN
SCIP_RETCODE SCIPsepalpCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to separate */
   SCIP_RESULT*          result              /**< pointer to store the result of the separation call */
   );

/** calls separation method of single constraint for given primal solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note This is an advanced method and should be used with caution.
 */
EXTERN
SCIP_RETCODE SCIPsepasolCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to separate */
   SCIP_SOL*             sol,                /**< primal solution that should be separated*/
   SCIP_RESULT*          result              /**< pointer to store the result of the separation call */
   );

/** calls domain propagation method of single constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note This is an advanced method and should be used with caution.
 */
EXTERN
SCIP_RETCODE SCIPpropCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to propagate */
   SCIP_PROPTIMING       proptiming,         /**< current point in the node solving loop */
   SCIP_RESULT*          result              /**< pointer to store the result of the callback method */
   );

/** resolves propagation conflict of single constraint
 *
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note This is an advanced method and should be used with caution.  It may only be called for constraints that were not
 *        added to SCIP beforehand.
 */
EXTERN
SCIP_RETCODE SCIPrespropCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to resolve conflict for */
   SCIP_VAR*             infervar,           /**< the conflict variable whose bound change has to be resolved */
   int                   inferinfo,          /**< the user information passed to the corresponding SCIPinferVarLbCons() or SCIPinferVarUbCons() call */
   SCIP_BOUNDTYPE        boundtype,          /**< the type of the changed bound (lower or upper bound) */
   SCIP_BDCHGIDX*        bdchgidx,           /**< the index of the bound change, representing the point of time where the change took place */
   SCIP_Real             relaxedbd,          /**< the relaxed bound which is sufficient to be explained */
   SCIP_RESULT*          result              /**< pointer to store the result of the callback method */
   );

/** presolves of single constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *
 *  @note This is an advanced method and should be used with caution.
 */
EXTERN
SCIP_RETCODE SCIPpresolCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to presolve */
   int                   nrounds,            /**< number of presolving rounds already done */
   int                   nnewfixedvars,      /**< number of variables fixed since the last call to the presolving method */
   int                   nnewaggrvars,       /**< number of variables aggregated since the last call to the presolving method */
   int                   nnewchgvartypes,    /**< number of variable type changes since the last call to the presolving method */
   int                   nnewchgbds,         /**< number of variable bounds tightened since the last call to the presolving method */
   int                   nnewholes,          /**< number of domain holes added since the last call to the presolving method */
   int                   nnewdelconss,       /**< number of deleted constraints since the last call to the presolving method */
   int                   nnewaddconss,       /**< number of added constraints since the last call to the presolving method */
   int                   nnewupgdconss,      /**< number of upgraded constraints since the last call to the presolving method */
   int                   nnewchgcoefs,       /**< number of changed coefficients since the last call to the presolving method */
   int                   nnewchgsides,       /**< number of changed left or right hand sides since the last call to the presolving method */
   int*                  nfixedvars,         /**< pointer to count total number of variables fixed of all presolvers */
   int*                  naggrvars,          /**< pointer to count total number of variables aggregated of all presolvers */
   int*                  nchgvartypes,       /**< pointer to count total number of variable type changes of all presolvers */
   int*                  nchgbds,            /**< pointer to count total number of variable bounds tightened of all presolvers */
   int*                  naddholes,          /**< pointer to count total number of domain holes added of all presolvers */
   int*                  ndelconss,          /**< pointer to count total number of deleted constraints of all presolvers */
   int*                  naddconss,          /**< pointer to count total number of added constraints of all presolvers */
   int*                  nupgdconss,         /**< pointer to count total number of upgraded constraints of all presolvers */
   int*                  nchgcoefs,          /**< pointer to count total number of changed coefficients of all presolvers */
   int*                  nchgsides,          /**< pointer to count total number of changed left/right hand sides of all presolvers */
   SCIP_RESULT*          result              /**< pointer to store the result of the callback method */
   );

/** calls constraint activation notification method of single constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *
 *  @note This is an advanced method and should be used with caution.  It may only be called for constraints that were not
 *      added to SCIP beforehand.
 */
EXTERN
SCIP_RETCODE SCIPactiveCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint to notify */
   );

/** calls constraint deactivation notification method of single constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note This is an advanced method and should be used with caution. It may only be called for constraints that were not
 *        added to SCIP beforehand.
 */
EXTERN
SCIP_RETCODE SCIPdeactiveCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint to notify */
   );

/** outputs constraint information to file stream via the message handler system
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note If the message handler is set to a NULL pointer nothing will be printed
 */
EXTERN
SCIP_RETCODE SCIPprintCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/** method to collect the variables of a constraint
 *
 *  If the number of variables is greater than the available slots in the variable array, nothing happens except that
 *  the success point is set to FALSE. With the method SCIPgetConsNVars() it is possible to get the number of variables
 *  a constraint has in its scope.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note The success pointer indicates if all variables were copied into the vars arrray.
 *
 *  @note It might be that a constraint handler does not support this functionality, in that case the success pointer is
 *        set to FALSE.
 */
EXTERN
SCIP_RETCODE SCIPgetConsVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint for which the variables are wanted */
   SCIP_VAR**            vars,               /**< array to store the involved variable of the constraint */
   int                   varssize,           /**< available slots in vars array which is needed to check if the array is large enough */
   SCIP_Bool*            success             /**< pointer to store whether the variables are successfully copied */
   );

/** methed to collect the number of variables of a constraint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  @note The success pointer indicates if the contraint handler was able to return the number of variables
 *
 *  @note It might be that a constraint handler does not support this functionality, in that case the success pointer is
 *        set to FALSE
 */
EXTERN
SCIP_RETCODE SCIPgetConsNVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint for which the number of variables is wanted */
   int*                  nvars,              /**< pointer to store the number of variables */
   SCIP_Bool*            success             /**< pointer to store whether the constraint successfully returned the number of variables */
   );

/**@} */




/*
 * LP methods
 */

/**@name LP Methods */
/**@{ */

/** returns, whether the LP was or is to be solved in the current node
 *
 *  @return whether the LP was or is to be solved in the current node.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Bool SCIPhasCurrentNodeLP(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns, whether the LP of the current node is already constructed
 *
 *  @return whether the LP of the current node is already constructed.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Bool SCIPisLPConstructed(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** makes sure that the LP of the current node is loaded and may be accessed through the LP information methods
 *
 *  @warning Contructing the LP might change the amount of variables known in the transformed problem and therefore also
 *           the variables array of SCIP (returned by SCIPgetVars() and SCIPgetVarsData()), so it might be necessary to
 *           call one of the later method after this one
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPconstructLP(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool*            cutoff              /**< pointer to store whether the node can be cut off */
   );

/** makes sure that the LP of the current node is flushed
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPflushLP(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets solution status of current LP
 *
 *  @return the solution status of current LP.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_LPSOLSTAT SCIPgetLPSolstat(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether the current lp is a relaxation of the current problem and its optimal objective value is a local lower bound
 *
 *  @return whether the current lp is a relaxation of the current problem and its optimal objective value is a local lower bound.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Bool SCIPisLPRelax(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets objective value of current LP (which is the sum of column and loose objective value)
 *
 *  @return the objective value of current LP (which is the sum of column and loose objective value).
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetLPObjval(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets part of objective value of current LP that results from COLUMN variables only
 *
 *  @return the part of objective value of current LP that results from COLUMN variables only.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetLPColumnObjval(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets part of objective value of current LP that results from LOOSE variables only
 *
 *  @return part of objective value of current LP that results from LOOSE variables only.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetLPLooseObjval(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the global pseudo objective value; that is all variables set to their best  (w.r.t. the objective
 *  function) global bound
 *
 *  @return the global pseudo objective value; that is all variables set to their best  (w.r.t. the objective
 *  function) global bound.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetGlobalPseudoObjval(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the pseudo objective value for the current search node; that is all variables set to their best (w.r.t. the
 *  objective function) local bound
 *
 *  @return the pseudo objective value for the current search node; that is all variables set to their best (w.r.t. the
 *  objective function) local bound.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetPseudoObjval(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether the root lp is a relaxation of the problem and its optimal objective value is a global lower bound
 *
 *  @return whether the root lp is a relaxation of the problem and its optimal objective value is a global lower bound.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Bool SCIPisRootLPRelax(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the objective value of the root node LP or SCIP_INVALID if the root node LP was not (yet) solved
 *
 *  @return the objective value of the root node LP or SCIP_INVALID if the root node LP was not (yet) solved.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetLPRootObjval(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets part of the objective value of the root node LP that results from COLUMN variables only;
 *  returns SCIP_INVALID if the root node LP was not (yet) solved
 *
 *  @return the part of the objective value of the root node LP that results from COLUMN variables only;
 *  or SCIP_INVALID if the root node LP was not (yet) solved.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetLPRootColumnObjval(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets part of the objective value of the root node LP that results from LOOSE variables only;
 *  returns SCIP_INVALID if the root node LP was not (yet) solved
 *
 *  @return the part of the objective value of the root node LP that results from LOOSE variables only;
 *  or SCIP_INVALID if the root node LP was not (yet) solved.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetLPRootLooseObjval(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current LP columns along with the current number of LP columns
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetLPColsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_COL***           cols,               /**< pointer to store the array of LP columns, or NULL */
   int*                  ncols               /**< pointer to store the number of LP columns, or NULL */
   );

/** gets current LP columns
 *
 *  @return the current LP columns.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_COL** SCIPgetLPCols(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current number of LP columns
 *
 *  @return the current number of LP columns.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNLPCols(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current LP rows along with the current number of LP rows
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetLPRowsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW***           rows,               /**< pointer to store the array of LP rows, or NULL */
   int*                  nrows               /**< pointer to store the number of LP rows, or NULL */
   );

/** gets current LP rows
 *
 *  @return the current LP rows.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_ROW** SCIPgetLPRows(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current number of LP rows
 *
 *  @return the current number of LP rows.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNLPRows(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns TRUE iff all columns, i.e. every variable with non-empty column w.r.t. all ever created rows, are present
 *  in the LP, and FALSE, if there are additional already existing columns, that may be added to the LP in pricing
 *
 *  @return TRUE iff all columns, i.e. every variable with non-empty column w.r.t. all ever created rows, are present
 *  in the LP, and FALSE, if there are additional already existing columns, that may be added to the LP in pricing.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Bool SCIPallColsInLP(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether the current LP solution is basic, i.e. is defined by a valid simplex basis
 *
 *  @return whether the current LP solution is basic, i.e. is defined by a valid simplex basis.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Bool SCIPisLPSolBasic(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets all indices of basic columns and rows: index i >= 0 corresponds to column i, index i < 0 to row -i-1
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetLPBasisInd(
   SCIP*                 scip,               /**< SCIP data structure */
   int*                  basisind            /**< pointer to store basis indices ready to keep number of rows entries */
   );

/** gets a row from the inverse basis matrix B^-1
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetLPBInvRow(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   r,                  /**< row number */
   SCIP_Real*            coef                /**< pointer to store the coefficients of the row */
   );

/** gets a column from the inverse basis matrix B^-1
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetLPBInvCol(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   c,                  /**< column number of B^-1; this is NOT the number of the column in the LP
                                              *   returned by SCIPcolGetLPPos(); you have to call SCIPgetBasisInd()
                                              *   to get the array which links the B^-1 column numbers to the row and
                                              *   column numbers of the LP! c must be between 0 and nrows-1, since the
                                              *   basis has the size nrows * nrows */
   SCIP_Real*            coef                /**< pointer to store the coefficients of the column */
   );

/** gets a row from the product of inverse basis matrix B^-1 and coefficient matrix A (i.e. from B^-1 * A)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetLPBInvARow(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   r,                  /**< row number */
   SCIP_Real*            binvrow,            /**< row in B^-1 from prior call to SCIPgetLPBInvRow(), or NULL */
   SCIP_Real*            coef                /**< pointer to store the coefficients of the row */
   );

/** gets a column from the product of inverse basis matrix B^-1 and coefficient matrix A (i.e. from B^-1 * A),
 *  i.e., it computes B^-1 * A_c with A_c being the c'th column of A
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetLPBInvACol(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   c,                  /**< column number which can be accessed by SCIPcolGetLPPos() */
   SCIP_Real*            coef                /**< pointer to store the coefficients of the column */
   );

/** calculates a weighted sum of all LP rows; for negative weights, the left and right hand side of the corresponding
 *  LP row are swapped in the summation
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPsumLPRows(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            weights,            /**< row weights in row summation */
   SCIP_REALARRAY*       sumcoef,            /**< array to store sum coefficients indexed by variables' probindex */
   SCIP_Real*            sumlhs,             /**< pointer to store the left hand side of the row summation */
   SCIP_Real*            sumrhs              /**< pointer to store the right hand side of the row summation */
   );

/** calculates a MIR cut out of the weighted sum of LP rows; The weights of modifiable rows are set to 0.0, because these
 *  rows cannot participate in a MIR cut.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcalcMIR(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< the solution that should be separated, or NULL for LP solution */
   SCIP_Real             boundswitch,        /**< fraction of domain up to which lower bound is used in transformation */
   SCIP_Bool             usevbds,            /**< should variable bounds be used in bound transformation? */
   SCIP_Bool             allowlocal,         /**< should local information allowed to be used, resulting in a local cut? */
   SCIP_Bool             fixintegralrhs,     /**< should complementation tried to be adjusted such that rhs gets fractional? */
   int*                  boundsfortrans,     /**< bounds that should be used for transformed variables: vlb_idx/vub_idx,  
                                              *   -1 for global lb/ub, -2 for local lb/ub, or -3 for using closest bound;
                                              *   NULL for using closest bound for all variables */
   SCIP_BOUNDTYPE*       boundtypesfortrans, /**< type of bounds that should be used for transformed variables; 
                                              *   NULL for using closest bound for all variables */
   int                   maxmksetcoefs,      /**< maximal number of nonzeros allowed in aggregated base inequality */
   SCIP_Real             maxweightrange,     /**< maximal valid range max(|weights|)/min(|weights|) of row weights */
   SCIP_Real             minfrac,            /**< minimal fractionality of rhs to produce MIR cut for */
   SCIP_Real             maxfrac,            /**< maximal fractionality of rhs to produce MIR cut for */
   SCIP_Real*            weights,            /**< row weights in row summation; some weights might be set to zero */
   int*                  sidetypes,          /**< specify row side type (-1 = lhs, 0 = unkown, 1 = rhs) or NULL for automatic choices */
   SCIP_Real             scale,              /**< additional scaling factor multiplied to all rows */
   SCIP_Real*            mksetcoefs,         /**< array to store mixed knapsack set coefficients: size nvars; or NULL */
   SCIP_Bool*            mksetcoefsvalid,    /**< pointer to store whether mixed knapsack set coefficients are valid; or NULL */
   SCIP_Real*            mircoef,            /**< array to store MIR coefficients: must be of size SCIPgetNVars() */
   SCIP_Real*            mirrhs,             /**< pointer to store the right hand side of the MIR row */
   SCIP_Real*            cutactivity,        /**< pointer to store the activity of the resulting cut */
   SCIP_Bool*            success,            /**< pointer to store whether the returned coefficients are a valid MIR cut */
   SCIP_Bool*            cutislocal,         /**< pointer to store whether the returned cut is only valid locally */
   int*                  cutrank             /**< pointer to store the rank of the returned cut; or NULL */
   );

/** calculates a strong CG cut out of the weighted sum of LP rows; The weights of modifiable rows are set to 0.0, because these
 *  rows cannot participate in a MIR cut.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcalcStrongCG(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             boundswitch,        /**< fraction of domain up to which lower bound is used in transformation */
   SCIP_Bool             usevbds,            /**< should variable bounds be used in bound transformation? */
   SCIP_Bool             allowlocal,         /**< should local information allowed to be used, resulting in a local cut? */
   int                   maxmksetcoefs,      /**< maximal number of nonzeros allowed in aggregated base inequality */
   SCIP_Real             maxweightrange,     /**< maximal valid range max(|weights|)/min(|weights|) of row weights */
   SCIP_Real             minfrac,            /**< minimal fractionality of rhs to produce strong CG cut for */
   SCIP_Real             maxfrac,            /**< maximal fractionality of rhs to produce strong CG cut for */
   SCIP_Real*            weights,            /**< row weights in row summation; some weights might be set to zero */
   SCIP_Real             scale,              /**< additional scaling factor multiplied to all rows */
   SCIP_Real*            mircoef,            /**< array to store strong CG coefficients: must be of size SCIPgetNVars() */
   SCIP_Real*            mirrhs,             /**< pointer to store the right hand side of the strong CG row */
   SCIP_Real*            cutactivity,        /**< pointer to store the activity of the resulting cut */
   SCIP_Bool*            success,            /**< pointer to store whether the returned coefficients are a valid strong CG cut */
   SCIP_Bool*            cutislocal,         /**< pointer to store whether the returned cut is only valid locally */
   int*                  cutrank             /**< pointer to store the rank of the returned cut; or NULL */
   );

/** writes current LP to a file
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPwriteLP(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename            /**< file name */
   );

/** writes MIP relaxation of the current branch-and-bound node to a file
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPwriteMIP(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< file name */
   SCIP_Bool             genericnames,       /**< should generic names like x_i and row_j be used in order to avoid
                                              *   troubles with reserved symbols? */
   SCIP_Bool             origobj             /**< should the original objective function be used? */
   );

/** gets the LP interface of SCIP;
 *  with the LPI you can use all of the methods defined in scip/lpi.h;
 *
 *  @warning You have to make sure, that the full internal state of the LPI does not change or is recovered completely
 *           after the end of the method that uses the LPI. In particular, if you manipulate the LP or its solution
 *           (e.g. by calling one of the SCIPlpiAdd...() or one of the SCIPlpiSolve...() methods), you have to check in
 *           advance with SCIPlpiWasSolved() whether the LP is currently solved. If this is the case, you have to make
 *           sure, the internal solution status is recovered completely at the end of your method. This can be achieved
 *           by getting the LPI state before applying any LPI manipulations with SCIPlpiGetState() and restoring it
 *           afterwards with SCIPlpiSetState() and SCIPlpiFreeState(). Additionally you have to resolve the LP with the
 *           appropriate SCIPlpiSolve...() call in order to reinstall the internal solution status.
 *
 *  @warning Make also sure, that all parameter values that you have changed are set back to their original values.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetLPI(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_LPI**            lpi                 /**< pointer to store the LP interface */
   );

/** Displays quality information about the current LP solution. An LP solution need to be available. Information printed
 *  is subject to what the LP solver supports
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_FREE
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 *
 *  @note The printing process is done via the message handler system.
 */
EXTERN
SCIP_RETCODE SCIPprintLPSolutionQuality(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/** compute relative interior point to current LP
 *  @see SCIPlpComputeRelIntPoint
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcomputeLPRelIntPoint(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             relaxrows,          /**< should the rows be relaxed */
   SCIP_Bool             inclobjcutoff,      /**< should a row for the objective cutoff be included */
   char                  normtype,           /**< which norm to use: 'o'ne-norm or 's'upremum-norm */
   SCIP_Real             timelimit,          /**< time limit for LP solver */
   int                   iterlimit,          /**< iteration limit for LP solver */
   SCIP_SOL**            point               /**< relative interior point on exit */
   );

/**@} */



/*
 * LP column methods
 */

/**@name LP Column Methods */
/**@{ */

/** returns the reduced costs of a column in the last (feasible) LP
 *
 *  @return the reduced costs of a column in the last (feasible) LP
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetColRedcost(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_COL*             col                 /**< LP column */
   );

/** returns the Farkas coefficient of a column in the last (infeasible) LP
 *
 *  @return the Farkas coefficient of a column in the last (infeasible) LP
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetColFarkasCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_COL*             col                 /**< LP column */
   );

/** marks a column to be not removable from the LP in the current node
 *
 *  @pre this method can be called in the following stage of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
void SCIPmarkColNotRemovableLocal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_COL*             col                 /**< LP column */
   );

/**@} */




/*
 * LP row methods
 */

/**@name LP Row Methods */
/**@{ */

/** creates and captures an LP row from a constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateRowCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row,                /**< pointer to row */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler that creates the row */
   const char*           name,               /**< name of row */
   int                   len,                /**< number of nonzeros in the row */
   SCIP_COL**            cols,               /**< array with columns of row entries */
   SCIP_Real*            vals,               /**< array with coefficients of row entries */
   SCIP_Real             lhs,                /**< left hand side of row */
   SCIP_Real             rhs,                /**< right hand side of row */
   SCIP_Bool             local,              /**< is row only valid locally? */
   SCIP_Bool             modifiable,         /**< is row modifiable during node processing (subject to column generation)? */
   SCIP_Bool             removable           /**< should the row be removed from the LP due to aging or cleanup? */
   );

/** creates and captures an LP row from a separator
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateRowSepa(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row,                /**< pointer to row */
   SCIP_SEPA*            sepa,               /**< separator that creates the row */
   const char*           name,               /**< name of row */
   int                   len,                /**< number of nonzeros in the row */
   SCIP_COL**            cols,               /**< array with columns of row entries */
   SCIP_Real*            vals,               /**< array with coefficients of row entries */
   SCIP_Real             lhs,                /**< left hand side of row */
   SCIP_Real             rhs,                /**< right hand side of row */
   SCIP_Bool             local,              /**< is row only valid locally? */
   SCIP_Bool             modifiable,         /**< is row modifiable during node processing (subject to column generation)? */
   SCIP_Bool             removable           /**< should the row be removed from the LP due to aging or cleanup? */
   );

/** creates and captures an LP row from an unspecified source
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateRowUnspec(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row,                /**< pointer to row */
   const char*           name,               /**< name of row */
   int                   len,                /**< number of nonzeros in the row */
   SCIP_COL**            cols,               /**< array with columns of row entries */
   SCIP_Real*            vals,               /**< array with coefficients of row entries */
   SCIP_Real             lhs,                /**< left hand side of row */
   SCIP_Real             rhs,                /**< right hand side of row */
   SCIP_Bool             local,              /**< is row only valid locally? */
   SCIP_Bool             modifiable,         /**< is row modifiable during node processing (subject to column generation)? */
   SCIP_Bool             removable           /**< should the row be removed from the LP due to aging or cleanup? */
   );

/** creates and captures an LP row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @deprecated Please use SCIPcreateRowCons() or SCIPcreateRowSepa() when calling from a constraint handler or separator in order
 *              to facilitate correct statistics. If the call is from neither a constraint handler or separator, use SCIPcreateRowUnspec().
 */
EXTERN
SCIP_DEPRECATED
SCIP_RETCODE SCIPcreateRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row,                /**< pointer to row */
   const char*           name,               /**< name of row */
   int                   len,                /**< number of nonzeros in the row */
   SCIP_COL**            cols,               /**< array with columns of row entries */
   SCIP_Real*            vals,               /**< array with coefficients of row entries */
   SCIP_Real             lhs,                /**< left hand side of row */
   SCIP_Real             rhs,                /**< right hand side of row */
   SCIP_Bool             local,              /**< is row only valid locally? */
   SCIP_Bool             modifiable,         /**< is row modifiable during node processing (subject to column generation)? */
   SCIP_Bool             removable           /**< should the row be removed from the LP due to aging or cleanup? */
   );

/** creates and captures an LP row without any coefficients from a constraint handler
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateEmptyRowCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row,                /**< pointer to row */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler that creates the row */
   const char*           name,               /**< name of row */
   SCIP_Real             lhs,                /**< left hand side of row */
   SCIP_Real             rhs,                /**< right hand side of row */
   SCIP_Bool             local,              /**< is row only valid locally? */
   SCIP_Bool             modifiable,         /**< is row modifiable during node processing (subject to column generation)? */
   SCIP_Bool             removable           /**< should the row be removed from the LP due to aging or cleanup? */
   );

/** creates and captures an LP row without any coefficients from a separator
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateEmptyRowSepa(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row,                /**< pointer to row */
   SCIP_SEPA*            sepa,               /**< separator that creates the row */
   const char*           name,               /**< name of row */
   SCIP_Real             lhs,                /**< left hand side of row */
   SCIP_Real             rhs,                /**< right hand side of row */
   SCIP_Bool             local,              /**< is row only valid locally? */
   SCIP_Bool             modifiable,         /**< is row modifiable during node processing (subject to column generation)? */
   SCIP_Bool             removable           /**< should the row be removed from the LP due to aging or cleanup? */
   );

/** creates and captures an LP row without any coefficients from an unspecified source
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateEmptyRowUnspec(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row,                /**< pointer to row */
   const char*           name,               /**< name of row */
   SCIP_Real             lhs,                /**< left hand side of row */
   SCIP_Real             rhs,                /**< right hand side of row */
   SCIP_Bool             local,              /**< is row only valid locally? */
   SCIP_Bool             modifiable,         /**< is row modifiable during node processing (subject to column generation)? */
   SCIP_Bool             removable           /**< should the row be removed from the LP due to aging or cleanup? */
   );

/** creates and captures an LP row without any coefficients
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @deprecated Please use SCIPcreateEmptyRowCons() or SCIPcreateEmptyRowSepa() when calling from a constraint handler or separator in order
 *              to facilitate correct statistics. If the call is from neither a constraint handler or separator, use SCIPcreateEmptyRowUnspec().
 */
EXTERN
SCIP_DEPRECATED
SCIP_RETCODE SCIPcreateEmptyRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row,                /**< pointer to row */
   const char*           name,               /**< name of row */
   SCIP_Real             lhs,                /**< left hand side of row */
   SCIP_Real             rhs,                /**< right hand side of row */
   SCIP_Bool             local,              /**< is row only valid locally? */
   SCIP_Bool             modifiable,         /**< is row modifiable during node processing (subject to column generation)? */
   SCIP_Bool             removable           /**< should the row be removed from the LP due to aging or cleanup? */
   );

/** increases usage counter of LP row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcaptureRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< row to capture */
   );

/** decreases usage counter of LP row, and frees memory if necessary
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPreleaseRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row                 /**< pointer to LP row */
   );

/** changes left hand side of LP row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgRowLhs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_Real             lhs                 /**< new left hand side */
   );

/** changes right hand side of LP row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgRowRhs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_Real             rhs                 /**< new right hand side */
   );

/** informs row, that all subsequent additions of variables to the row should be cached and not directly applied;
 *  after all additions were applied, SCIPflushRowExtensions() must be called;
 *  while the caching of row extensions is activated, information methods of the row give invalid results;
 *  caching should be used, if a row is build with SCIPaddVarToRow() calls variable by variable to increase
 *  the performance
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcacheRowExtensions(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** flushes all cached row extensions after a call of SCIPcacheRowExtensions() and merges coefficients with
 *  equal columns into a single coefficient
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPflushRowExtensions(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** resolves variable to columns and adds them with the coefficient to the row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note In case calling this method in the enforcement process of an lp solution, it might be that some variables,
 *        that were not yet in the LP (e.g. dynamic columns) will change their lp solution value returned by SCIP.
 *        For example, a variable, which has a negative objective value, that has no column in the lp yet, is in the lp solution
 *        on its upper bound (variables with status SCIP_VARSTATUS_LOOSE are in an lp solution on it's best bound), but
 *        creating the column, changes the solution value (variable than has status SCIP_VARSTATUS_COLUMN, and the
 *        initialization sets the lp solution value) to 0.0. (This leads to the conclusion that, if a constraint was
 *        violated, the linear relaxation might not be violated anymore.)
 */
EXTERN
SCIP_RETCODE SCIPaddVarToRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             val                 /**< value of coefficient */
   );

/** resolves variables to columns and adds them with the coefficients to the row;
 *  this method caches the row extensions and flushes them afterwards to gain better performance
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddVarsToRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   int                   nvars,              /**< number of variables to add to the row */
   SCIP_VAR**            vars,               /**< problem variables to add */
   SCIP_Real*            vals                /**< values of coefficients */
   );

/** resolves variables to columns and adds them with the same single coefficient to the row;
 *  this method caches the row extensions and flushes them afterwards to gain better performance
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddVarsToRowSameCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   int                   nvars,              /**< number of variables to add to the row */
   SCIP_VAR**            vars,               /**< problem variables to add */
   SCIP_Real             val                 /**< unique value of all coefficients */
   );

/** tries to find a value, such that all row coefficients, if scaled with this value become integral
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcalcRowIntegralScalar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_Real             mindelta,           /**< minimal relative allowed difference of scaled coefficient s*c and integral i */
   SCIP_Real             maxdelta,           /**< maximal relative allowed difference of scaled coefficient s*c and integral i */
   SCIP_Longint          maxdnom,            /**< maximal denominator allowed in rational numbers */
   SCIP_Real             maxscale,           /**< maximal allowed scalar */
   SCIP_Bool             usecontvars,        /**< should the coefficients of the continuous variables also be made integral? */
   SCIP_Real*            intscalar,          /**< pointer to store scalar that would make the coefficients integral, or NULL */
   SCIP_Bool*            success             /**< stores whether returned value is valid */
   );

/** tries to scale row, s.t. all coefficients (of integer variables) become integral
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPmakeRowIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_Real             mindelta,           /**< minimal relative allowed difference of scaled coefficient s*c and integral i */
   SCIP_Real             maxdelta,           /**< maximal relative allowed difference of scaled coefficient s*c and integral i */
   SCIP_Longint          maxdnom,            /**< maximal denominator allowed in rational numbers */
   SCIP_Real             maxscale,           /**< maximal value to scale row with */
   SCIP_Bool             usecontvars,        /**< should the coefficients of the continuous variables also be made integral? */
   SCIP_Bool*            success             /**< stores whether row could be made rational */
   );

/** marks a row to be not removable from the LP in the current node
 *
 *  @pre this method can be called in the following stage of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
void SCIPmarkRowNotRemovableLocal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** returns minimal absolute value of row vector's non-zero coefficients
 *
 *  @return minimal absolute value of row vector's non-zero coefficients
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowMinCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** returns maximal absolute value of row vector's non-zero coefficients
 *
 *  @return maximal absolute value of row vector's non-zero coefficients
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowMaxCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** returns the minimal activity of a row w.r.t. the column's bounds
 *
 *  @return the minimal activity of a row w.r.t. the column's bounds
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowMinActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** returns the maximal activity of a row w.r.t. the column's bounds
 *
 *  @return the maximal activity of a row w.r.t. the column's bounds
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowMaxActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** recalculates the activity of a row in the last LP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPrecalcRowLPActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** returns the activity of a row in the last LP solution
 *
 *  @return activity of a row in the last LP solution
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowLPActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** returns the feasibility of a row in the last LP solution
 *
 *  @return the feasibility of a row in the last LP solution: negative value means infeasibility
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowLPFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** recalculates the activity of a row for the current pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPrecalcRowPseudoActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** returns the activity of a row for the current pseudo solution
 *
 *  @return the activity of a row for the current pseudo solution
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowPseudoActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** returns the feasibility of a row for the current pseudo solution: negative value means infeasibility
 *
 *  @return the feasibility of a row for the current pseudo solution: negative value means infeasibility
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowPseudoFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** recalculates the activity of a row in the last LP or pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPrecalcRowActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** returns the activity of a row in the last LP or pseudo solution
 *
 *  @return the activity of a row in the last LP or pseudo solution
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** returns the feasibility of a row in the last LP or pseudo solution
 *
 *  @return the feasibility of a row in the last LP or pseudo solution
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   );

/** returns the activity of a row for the given primal solution
 *
 *  @return the activitiy of a row for the given primal solution
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowSolActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_SOL*             sol                 /**< primal CIP solution */
   );

/** returns the feasibility of a row for the given primal solution
 *
 *  @return the feasibility of a row for the given primal solution
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetRowSolFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_SOL*             sol                 /**< primal CIP solution */
   );

/** output row to file stream via the message handler system
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre this method can be called in one of the following stages of the SCIP solving process:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPprintRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/**@} */


/*
 * NLP methods
 */

/**@name NLP Methods */
/**@{ */

/** returns whether the NLP relaxation has been enabled
 *
 * If the NLP relaxation is enabled, then SCIP will construct the NLP relaxation when the solving process is about to begin.
 * To check whether an NLP is existing, use SCIPisNLPConstructed().
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 * @see SCIPenableNLP
 */
EXTERN
SCIP_Bool SCIPisNLPEnabled(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** marks that there are constraints that are representable by nonlinear rows
 *
 * This method should be called by a constraint handler if it has constraints that have a representation as nonlinear rows.
 *
 * The function should be called before the branch-and-bound process is initialized, e.g., when presolve is exiting.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
void SCIPenableNLP(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns, whether an NLP has been constructed
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Bool SCIPisNLPConstructed(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether the NLP has a continuous variable in a nonlinear term
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Bool SCIPhasNLPContinuousNonlinearity(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current NLP variables along with the current number of NLP variables
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNLPVarsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           vars,               /**< pointer to store the array of NLP variables, or NULL */
   int*                  nvars               /**< pointer to store the number of NLP variables, or NULL */
   );

/** gets array with variables of the NLP
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_VAR** SCIPgetNLPVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current number of variables in NLP
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetNNLPVars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** computes for each variables the number of NLP rows in which the variable appears in a nonlinear var
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNLPVarsNonlinearity(
   SCIP*                 scip,               /**< SCIP data structure */
   int*                  nlcount             /**< an array of length at least SCIPnlpGetNVars() to store nonlinearity counts of variables */
   );

/** returns dual solution values associated with lower bounds of NLP variables
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real* SCIPgetNLPVarsLbDualsol(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns dual solution values associated with upper bounds of NLP variables
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real* SCIPgetNLPVarsUbDualsol(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current NLP nonlinear rows along with the current number of NLP nonlinear rows
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNLPNlRowsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW***         nlrows,             /**< pointer to store the array of NLP nonlinear rows, or NULL */
   int*                  nnlrows             /**< pointer to store the number of NLP nonlinear rows, or NULL */
   );

/** gets array with nonlinear rows of the NLP
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NLROW** SCIPgetNLPNlRows(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current number of nonlinear rows in NLP
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetNNLPNlRows(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** adds a nonlinear row to the NLP. This row is captured by the NLP.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow               /**< nonlinear row to add to NLP */
   );

/** makes sure that the NLP of the current node is flushed
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPflushNLP(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets or clears initial primal guess for NLP solution (start point for NLP solver)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetNLPInitialGuess(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            initialguess        /**< values of initial guess (corresponding to variables from SCIPgetNLPVarsData), or NULL to use no start point */
   );

/** sets initial primal guess for NLP solution (start point for NLP solver)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetNLPInitialGuessSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< solution which values should be taken as initial guess, or NULL for LP solution */
   );

/** solves the current NLP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsolveNLP(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets solution status of current NLP
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NLPSOLSTAT SCIPgetNLPSolstat(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets termination status of last NLP solve
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NLPTERMSTAT SCIPgetNLPTermstat(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gives statistics (number of iterations, solving time, ...) of last NLP solve
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNLPStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPSTATISTICS*   statistics          /**< pointer to store statistics */
   );

/** gets objective value of current NLP
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetNLPObjval(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** indicates whether a feasible solution for the current NLP is available
 * thus, returns whether the solution status <= feasible
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Bool SCIPhasNLPSolution(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets fractional variables of last NLP solution along with solution values and fractionalities
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNLPFracVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           fracvars,           /**< pointer to store the array of NLP fractional variables, or NULL */
   SCIP_Real**           fracvarssol,        /**< pointer to store the array of NLP fractional variables solution values, or NULL */
   SCIP_Real**           fracvarsfrac,       /**< pointer to store the array of NLP fractional variables fractionalities, or NULL */
   int*                  nfracvars,          /**< pointer to store the number of NLP fractional variables , or NULL */
   int*                  npriofracvars       /**< pointer to store the number of NLP fractional variables with maximal branching priority, or NULL */
   );

/** gets integer parameter of NLP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNLPIntPar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPPARAM         type,               /**< parameter number */
   int*                  ival                /**< pointer to store the parameter value */
   );

/** sets integer parameter of NLP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetNLPIntPar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPPARAM         type,               /**< parameter number */
   int                   ival                /**< parameter value */
   );

/** gets floating point parameter of NLP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNLPRealPar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPPARAM         type,               /**< parameter number */
   SCIP_Real*            dval                /**< pointer to store the parameter value */
   );

/** sets floating point parameter of NLP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetNLPRealPar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPPARAM         type,               /**< parameter number */
   SCIP_Real             dval                /**< parameter value */
   );

/** gets string parameter of NLP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNLPStringPar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPPARAM         type,               /**< parameter number */
   const char**          sval                /**< pointer to store the parameter value */
   );

/** sets string parameter of NLP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetNLPStringPar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPPARAM         type,               /**< parameter number */
   const char*           sval                /**< parameter value */
   );

/** writes current NLP to a file
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPwriteNLP(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename            /**< file name */
   );

/** gets the NLP interface and problem used by the SCIP NLP;
 *  with the NLPI and its problem you can use all of the methods defined in nlpi/nlpi.h;
 *
 *  @warning You have to make sure, that the full internal state of the NLPI does not change or is recovered completely
 *           after the end of the method that uses the NLPI. In particular, if you manipulate the NLP or its solution
 *           (e.g. by calling one of the SCIPnlpiAdd...() or the SCIPnlpiSolve() method), you have to check in advance
 *           whether the NLP is currently solved.  If this is the case, you have to make sure, the internal solution
 *           status is recovered completely at the end of your method. Additionally you have to resolve the NLP with
 *           SCIPnlpiSolve() in order to reinstall the internal solution status.
 *
 *  @warning Make also sure, that all parameter values that you have changed are set back to their original values.
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNLPI(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLPI**           nlpi,               /**< pointer to store the NLP solver interface */
   SCIP_NLPIPROBLEM**    nlpiproblem         /**< pointer to store the NLP solver interface problem */
   );

/**@} */


/*
 * NLP diving methods
 */

/**@name NLP Diving Methods */
/**@{ */

/** initiates NLP diving
 * making methods SCIPchgVarObjDiveNLP(), SCIPchgVarBoundsDiveNLP(), SCIPchgVarsBoundsDiveNLP(), and SCIPsolveDiveNLP() available
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPstartDiveNLP(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** ends NLP diving
 * resets changes made by SCIPchgVarObjDiveNLP(), SCIPchgVarBoundsDiveNLP(), and SCIPchgVarsBoundsDiveNLP()
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPendDiveNLP(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** changes linear objective coefficient of a variable in diving NLP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgVarObjDiveNLP(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable which coefficient to change */
   SCIP_Real             coef                /**< new value for coefficient */
   );

/** changes bounds of a variable in diving NLP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgVarBoundsDiveNLP(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable which bounds to change */
   SCIP_Real             lb,                 /**< new lower bound */
   SCIP_Real             ub                  /**< new upper bound */
   );

/** changes bounds of a set of variables in diving NLP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgVarsBoundsDiveNLP(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nvars,              /**< number of variables which bounds to changes */
   SCIP_VAR**            vars,               /**< variables which bounds to change */
   SCIP_Real*            lbs,                /**< new lower bounds */
   SCIP_Real*            ubs                 /**< new upper bounds */
   );

/** solves diving NLP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsolveDiveNLP(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@} */


/*
 * NLP nonlinear row methods
 */

/**@name NLP Nonlinear Row Methods */
/**@{ */

/** creates and captures an NLP row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW**          nlrow,              /**< buffer to store pointer to nonlinear row */
   const char*           name,               /**< name of nonlinear row */
   SCIP_Real             constant,           /**< constant */
   int                   nlinvars,           /**< number of linear variables */
   SCIP_VAR**            linvars,            /**< linear variables, or NULL if nlinvars == 0 */
   SCIP_Real*            lincoefs,           /**< linear coefficients, or NULL if nlinvars == 0 */
   int                   nquadvars,          /**< number of variables in quadratic term */
   SCIP_VAR**            quadvars,           /**< variables in quadratic terms, or NULL if nquadvars == 0 */
   int                   nquadelems,         /**< number of elements in quadratic term */
   SCIP_QUADELEM*        quadelems,          /**< elements (i.e., monomials) in quadratic term, or NULL if nquadelems == 0 */
   SCIP_EXPRTREE*        expression,         /**< nonlinear expression, or NULL */
   SCIP_Real             lhs,                /**< left hand side */
   SCIP_Real             rhs                 /**< right hand side */
   );

/** creates and captures an NLP nonlinear row without any coefficients
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateEmptyNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW**          nlrow,              /**< pointer to nonlinear row */
   const char*           name,               /**< name of nonlinear row */
   SCIP_Real             lhs,                /**< left hand side */
   SCIP_Real             rhs                 /**< right hand side */
   );

/** creates and captures an NLP row from a linear row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateNlRowFromRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW**          nlrow,              /**< pointer to nonlinear row */
   SCIP_ROW*             row                 /**< the linear row to copy */
   );

/** increases usage counter of NLP nonlinear row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcaptureNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow               /**< nonlinear row to capture */
   );

/** decreases usage counter of NLP nonlinear row, and frees memory if necessary
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPreleaseNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW**          nlrow               /**< pointer to nonlinear row */
   );

/** changes left hand side of NLP nonlinear row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgNlRowLhs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP nonlinear row */
   SCIP_Real             lhs                 /**< new left hand side */
   );

/** changes right hand side of NLP nonlinear row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgNlRowRhs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP nonlinear row */
   SCIP_Real             rhs                 /**< new right hand side */
   );

/** changes constant of NLP nonlinear row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgNlRowConstant(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   SCIP_Real             constant            /**< new value for constant */
   );

/** adds variable with a linear coefficient to the nonlinear row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddLinearCoefToNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             val                 /**< value of coefficient in linear part of row */
   );

/** adds variables with linear coefficients to the row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddLinearCoefsToNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   int                   nvars,              /**< number of variables to add to the row */
   SCIP_VAR**            vars,               /**< problem variables to add */
   SCIP_Real*            vals                /**< values of coefficients in linear part of row */
   );

/** changes linear coefficient of a variables in a row
 * setting the coefficient to 0.0 means that it is removed from the row
 * the variable does not need to exists before
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgNlRowLinearCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   SCIP_VAR*             var,                /**< variable */
   SCIP_Real             coef                /**< new value of coefficient */
   );

/** adds quadratic variable to the nonlinear row
 * after adding a quadratic variable, it can be used to add quadratic elements
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddQuadVarToNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   SCIP_VAR*             var                 /**< problem variable */
   );

/** adds quadratic variables to the nonlinear row
 * after adding quadratic variables, they can be used to add quadratic elements
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddQuadVarsToNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   int                   nvars,              /**< number of problem variables */
   SCIP_VAR**            vars                /**< problem variables */
   );

/** add a quadratic element to the nonlinear row
 * variable indices of the quadratic element need to be relative to quadratic variables array of row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddQuadElementToNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   SCIP_QUADELEM         quadelem            /**< quadratic element */
   );

/** adds quadratic elements to the nonlinear row
 * variable indices of the quadratic elements need to be relative to quadratic variables array of row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddQuadElementsToNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   int                   nquadelems,         /**< number of quadratic elements */
   SCIP_QUADELEM*        quadelems           /**< quadratic elements */
   );

/** changes coefficient in quadratic part of a row
 * setting the coefficient in the quadelement to 0.0 means that it is removed from the row
 * the element does not need to exists before
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgNlRowQuadElement(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   SCIP_QUADELEM         quadelement         /**< new quadratic element, or update for existing one */
   );

/** sets or deletes expression tree in the nonlinear row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetNlRowExprtree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   SCIP_EXPRTREE*        exprtree            /**< expression tree, or NULL */
   );

/** sets a parameter of expression tree in the nonlinear row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetNlRowExprtreeParam(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   int                   paramidx,           /**< index of parameter in expression tree */
   SCIP_Real             paramval            /**< new value of parameter in expression tree */
   );

/** sets parameters of expression tree in the nonlinear row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsetNlRowExprtreeParams(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   SCIP_Real*            paramvals           /**< new values of parameter in expression tree */
   );

/** recalculates the activity of a nonlinear row in the last NLP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPrecalcNlRowNLPActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow               /**< NLP nonlinear row */
   );

/** returns the activity of a nonlinear row in the last NLP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNlRowNLPActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP nonlinear row */
   SCIP_Real*            activity            /**< pointer to store activity value */
   );

/** gives the feasibility of a nonlinear row in the last NLP solution: negative value means infeasibility
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNlRowNLPFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP nonlinear row */
   SCIP_Real*            feasibility         /**< pointer to store feasibility value */
   );

/** recalculates the activity of a nonlinear row for the current pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPrecalcNlRowPseudoActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow               /**< NLP nonlinear row */
   );

/** gives the activity of a nonlinear row for the current pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNlRowPseudoActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP nonlinear row */
   SCIP_Real*            pseudoactivity      /**< pointer to store pseudo activity value */
   );

/** gives the feasibility of a nonlinear row for the current pseudo solution: negative value means infeasibility
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNlRowPseudoFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP nonlinear row */
   SCIP_Real*            pseudofeasibility   /**< pointer to store pseudo feasibility value */
   );

/** recalculates the activity of a nonlinear row in the last NLP or pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPrecalcNlRowActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow               /**< NLP nonlinear row */
   );

/** gives the activity of a nonlinear row in the last NLP or pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNlRowActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP nonlinear row */
   SCIP_Real*            activity            /**< pointer to store activity value */
   );

/** gives the feasibility of a nonlinear row in the last NLP or pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNlRowFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP nonlinear row */
   SCIP_Real*            feasibility         /**< pointer to store feasibility value */
   );

/** gives the activity of a nonlinear row for the given primal solution or NLP solution or pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNlRowSolActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP nonlinear row */
   SCIP_SOL*             sol,                /**< primal CIP solution, or NULL for NLP solution of pseudo solution */
   SCIP_Real*            activity            /**< pointer to store activity value */
   );

/** gives the feasibility of a nonlinear row for the given primal solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNlRowSolFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP nonlinear row */
   SCIP_SOL*             sol,                /**< primal CIP solution */
   SCIP_Real*            feasibility         /**< pointer to store feasibility value */
   );

/** gives the minimal and maximal activity of a nonlinear row w.r.t. the variable's bounds
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetNlRowActivityBounds(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   SCIP_Real*            minactivity,        /**< buffer to store minimal activity, or NULL */
   SCIP_Real*            maxactivity         /**< buffer to store maximal activity, or NULL */
   );

/** output nonlinear row to file stream
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPprintNlRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NLROW*           nlrow,              /**< NLP row */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/**@} */

/**@name Expression tree methods */
/**@{ */

/** replaces array of variables in expression tree by corresponding transformed variables
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetExprtreeTransformedVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EXPRTREE*        tree                /**< expression tree */
   );

/** evaluates an expression tree for a primal solution or LP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPevalExprtreeSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EXPRTREE*        tree,               /**< expression tree */
   SCIP_SOL*             sol,                /**< a solution, or NULL for current LP solution */
   SCIP_Real*            val                 /**< buffer to store value */
   );

/** evaluates an expression tree w.r.t. current global bounds
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPevalExprtreeGlobalBounds(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EXPRTREE*        tree,               /**< expression tree */
   SCIP_Real             infinity,           /**< value to use for infinity */
   SCIP_INTERVAL*        val                 /**< buffer to store result */
   );

/** evaluates an expression tree w.r.t. current local bounds
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPevalExprtreeLocalBounds(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EXPRTREE*        tree,               /**< expression tree */
   SCIP_Real             infinity,           /**< value to use for infinity */
   SCIP_INTERVAL*        val                 /**< buffer to store result */
   );

/**@} */


/*
 * cutting plane methods
 */

/**@name Cutting Plane Methods */
/**@{ */

/** returns efficacy of the cut with respect to the given primal solution or the current LP solution:
 *  e = -feasibility/norm
 *
 *  @return the efficacy of the cut with respect to the given primal solution or the current LP solution:
 *          e = -feasibility/norm
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Real SCIPgetCutEfficacy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution, or NULL for current LP solution */
   SCIP_ROW*             cut                 /**< separated cut */
   );

/** returns whether the cut's efficacy with respect to the given primal solution or the current LP solution is greater
 *  than the minimal cut efficacy
 *
 *  @return TRUE if the cut's efficacy with respect to the given primal solution or the current LP solution is greater
 *          than the minimal cut efficacy, otherwise FALSE
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Bool SCIPisCutEfficacious(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution, or NULL for current LP solution */
   SCIP_ROW*             cut                 /**< separated cut */
   );

/** checks, if the given cut's efficacy is larger than the minimal cut efficacy
 *
 *  @return TRUE if the given cut's efficacy is larger than the minimal cut efficacy, otherwise FALSE
 */
EXTERN
SCIP_Bool SCIPisEfficacious(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             efficacy            /**< efficacy of the cut */
   );

/** calculates the efficacy norm of the given vector, which depends on the "separating/efficacynorm" parameter
 *
 *  @return the efficacy norm of the given vector, which depends on the "separating/efficacynorm" parameter
 */
EXTERN
SCIP_Real SCIPgetVectorEfficacyNorm(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            vals,               /**< array of values */
   int                   nvals               /**< number of values */
   );

/** adds cut to separation storage
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddCut(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution that was separated, or NULL for LP solution */
   SCIP_ROW*             cut,                /**< separated cut */
   SCIP_Bool             forcecut,           /**< should the cut be forced to enter the LP? */
   SCIP_Bool*            infeasible          /**< pointer to store whether cut has been detected to be infeasible for local bounds */
   );

/** if not already existing, adds row to global cut pool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddPoolCut(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< cutting plane to add */
   );

/** removes the row from the global cut pool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPdelPoolCut(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< cutting plane to add */
   );

/** gets current cuts in the global cut pool
 *
 *  @return the current cuts in the global cut pool
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_CUT** SCIPgetPoolCuts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current number of rows in the global cut pool
 *
 *  @return the current number of rows in the global cut pool
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetNPoolCuts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the global cut pool used by SCIP
 *
 *  @return the global cut pool used by SCIP
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_CUTPOOL* SCIPgetGlobalCutpool(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** creates a cut pool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL**        cutpool,            /**< pointer to store cut pool */
   int                   agelimit            /**< maximum age a cut can reach before it is deleted from the pool */
   );

/** frees a cut pool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPfreeCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL**        cutpool             /**< pointer to store cut pool */
   );

/** if not already existing, adds row to a cut pool and captures it
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddRowCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL*         cutpool,            /**< cut pool */
   SCIP_ROW*             row                 /**< cutting plane to add */
   );

/** adds row to a cut pool and captures it; doesn't check for multiple cuts
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddNewRowCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL*         cutpool,            /**< cut pool */
   SCIP_ROW*             row                 /**< cutting plane to add */
   );

/** removes the LP row from a cut pool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPdelRowCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL*         cutpool,            /**< cut pool */
   SCIP_ROW*             row                 /**< row to remove */
   );

/** separates cuts from a cut pool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPseparateCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL*         cutpool,            /**< cut pool */
   SCIP_RESULT*          result              /**< pointer to store the result of the separation call */
   );

/** separates cuts w.r.t. given solution from a cut pool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPseparateSolCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL*         cutpool,            /**< cut pool */
   SCIP_SOL*             sol,                /**< solution to be separated */
   SCIP_RESULT*          result              /**< pointer to store the result of the separation call */
   );

/** if not already existing, adds row to the delayed global cut pool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is the stages \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddDelayedPoolCut(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< cutting plane to add */
   );

/** removes the row from the delayed global cut pool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is the stages \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPdelDelayedPoolCut(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< cutting plane to add */
   );

/** gets current cuts in the delayed global cut pool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is the stages \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_CUT** SCIPgetDelayedPoolCuts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current number of rows in the delayed global cut pool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is the stages \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetNDelayedPoolCuts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the delayed global cut pool used by SCIP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is the stages \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_CUTPOOL* SCIPgetDelayedGlobalCutpool(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** separates the given primal solution or the current LP solution by calling the separators and constraint handlers'
 *  separation methods;
 *  the generated cuts are stored in the separation storage and can be accessed with the methods SCIPgetCuts() and
 *  SCIPgetNCuts();
 *  after evaluating the cuts, you have to call SCIPclearCuts() in order to remove the cuts from the
 *  separation storage;
 *  it is possible to call SCIPseparateSol() multiple times with different solutions and evaluate the found cuts
 *  afterwards
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPseparateSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution that should be separated, or NULL for LP solution */
   SCIP_Bool             pretendroot,        /**< should the cut separators be called as if we are at the root node? */
   SCIP_Bool             onlydelayed,        /**< should only separators be called that were delayed in the previous round? */
   SCIP_Bool*            delayed,            /**< pointer to store whether a separator was delayed */
   SCIP_Bool*            cutoff              /**< pointer to store whether the node can be cut off */
   );

/** gets the array of cuts currently stored in the separation storage
 *
 *  @return the array of cuts currently stored in the separation storage
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_ROW** SCIPgetCuts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** get current number of cuts in the separation storage
 *
 *  @return the current number of cuts in the separation storage
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNCuts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** clears the separation storage
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPclearCuts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** removes cuts that are inefficacious w.r.t. the current LP solution from separation storage without adding the cuts to the LP
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPremoveInefficaciousCuts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@} */




/*
 * LP diving methods
 */

/**@name LP Diving Methods */
/**@{ */

/** initiates LP diving, making methods SCIPchgVarObjDive(), SCIPchgVarLbDive(), and SCIPchgVarUbDive() available
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 *
 *  @note diving is allowed even if the current LP is not flushed, not solved, or not solved to optimality; be aware
 *  that solving the (first) diving LP may take longer than expect and that the latter two cases could stem from
 *  numerical troubles during the last LP solve; because of this, most users will want to call this method only if
 *  SCIPgetLPSolstat(scip) == SCIP_LPSOLSTAT_OPTIMAL
 */
EXTERN
SCIP_RETCODE SCIPstartDive(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** quits LP diving and resets bounds and objective values of columns to the current node's values
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPendDive(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** changes cutoffbound in current dive
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPchgCutoffboundDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             newcutoffbound      /**< new cutoffbound */
   );

/** changes variable's objective value in current dive
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPchgVarObjDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the objective value for */
   SCIP_Real             newobj              /**< new objective value */
   );

/** changes variable's lower bound in current dive
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPchgVarLbDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   );

/** changes variable's upper bound in current dive
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPchgVarUbDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   );

/** adds a row to the LP in current dive
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPaddRowDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< row to be added */
   );

/** changes row lhs in current dive, change will be undone after diving ends, for permanent changes use SCIPchgRowLhs()
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPchgRowLhsDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< row to change the lhs for */
   SCIP_Real             newlhs              /**< new value for lhs */
   );

/** changes row rhs in current dive, change will be undone after diving ends, for permanent changes use SCIPchgRowRhs()
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPchgRowRhsDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< row to change the lhs for */
   SCIP_Real             newrhs              /**< new value for rhs */
   );

/** gets variable's objective value in current dive
 *
 *  @return the variable's objective value in current dive.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetVarObjDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get the bound for */
   );

/** gets variable's lower bound in current dive
 *
 *  @return the variable's lower bound in current dive.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetVarLbDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get the bound for */
   );

/** gets variable's upper bound in current dive
 *
 *  @return the variable's upper bound in current dive.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetVarUbDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get the bound for */
   );
/** solves the LP of the current dive; no separation or pricing is applied
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 *
 *  @note be aware that the LP solve may take longer than expected if SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL,
 *  compare the explanation of SCIPstartDive()
 */
EXTERN
SCIP_RETCODE SCIPsolveDiveLP(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   itlim,              /**< maximal number of LP iterations to perform, or -1 for no limit */
   SCIP_Bool*            lperror,            /**< pointer to store whether an unresolved LP error occurred */
   SCIP_Bool*            cutoff              /**< pointer to store whether the probing LP was infeasible or the objective
                                              *   limit was reached (or NULL, if not needed) */
   );

/** returns the number of the node in the current branch and bound run, where the last LP was solved in diving
 *  or probing mode
 *
 *  @return the number of the node in the current branch and bound run, where the last LP was solved in diving
 *  or probing mode.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Longint SCIPgetLastDivenode(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether we are in diving mode
 *
 *  @return whether we are in diving mode.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Bool SCIPinDive(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@} */




/*
 * probing methods
 */

/**@name Probing Methods */
/**@{ */

/** returns whether we are in probing mode; probing mode is activated via SCIPstartProbing() and stopped
 *  via SCIPendProbing()
 *
 *  @return TRUE, if SCIP is currently in probing mode, otherwise FALSE
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Bool SCIPinProbing(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** initiates probing, making methods SCIPnewProbingNode(), SCIPbacktrackProbing(), SCIPchgVarLbProbing(),
 *  SCIPchgVarUbProbing(), SCIPfixVarProbing(), SCIPpropagateProbing(), and SCIPsolveProbingLP() available
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note The collection of variable statistics is turned off during probing. If these statistics should be collected
 *        during probing use the method SCIPenableVarHistory() to turn the collection explicitly on.
 */
EXTERN
SCIP_RETCODE SCIPstartProbing(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** creates a new probing sub node, whose changes can be undone by backtracking to a higher node in the probing path
 *  with a call to SCIPbacktrackProbing();
 *  using a sub node for each set of probing bound changes can improve conflict analysis
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPnewProbingNode(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the current probing depth
 *
 *  @return the probing depth, i.e. the number of probing sub nodes existing in the probing path
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetProbingDepth(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** undoes all changes to the problem applied in probing up to the given probing depth;
 *  the changes of the probing node of the given probing depth are the last ones that remain active;
 *  changes that were applied before calling SCIPnewProbingNode() cannot be undone
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPbacktrackProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   probingdepth        /**< probing depth of the node in the probing path that should be reactivated */
   );

/** quits probing and resets bounds and constraints to the focus node's environment
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPendProbing(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** injects a change of variable's lower bound into current probing node; the same can also be achieved with a call to
 *  SCIPchgVarLb(), but in this case, the bound change would be treated like a deduction instead of a branching decision
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgVarLbProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   );

/** injects a change of variable's upper bound into current probing node; the same can also be achieved with a call to
 *  SCIPchgVarUb(), but in this case, the bound change would be treated like a deduction instead of a branching decision
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPchgVarUbProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   );

/** injects a change of variable's bounds into current probing node to fix the variable to the specified value;
 *  the same can also be achieved with a call to SCIPfixVar(), but in this case, the bound changes would be treated
 *  like deductions instead of branching decisions
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPfixVarProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             fixedval            /**< value to fix variable to */
   );

/** applies domain propagation on the probing sub problem, that was changed after SCIPstartProbing() was called;
 *  the propagated domains of the variables can be accessed with the usual bound accessing calls SCIPvarGetLbLocal()
 *  and SCIPvarGetUbLocal(); the propagation is only valid locally, i.e. the local bounds as well as the changed
 *  bounds due to SCIPchgVarLbProbing(), SCIPchgVarUbProbing(), and SCIPfixVarProbing() are used for propagation
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPpropagateProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   maxproprounds,      /**< maximal number of propagation rounds (-1: no limit, 0: parameter settings) */
   SCIP_Bool*            cutoff,             /**< pointer to store whether the probing node can be cut off */
   SCIP_Longint*         ndomredsfound       /**< pointer to store the number of domain reductions found, or NULL */
   );

/** applies domain propagation on the probing sub problem, that was changed after SCIPstartProbing() was called;
 *  only propagations of the binary variables fixed at the current probing node that are triggered by the implication
 *  graph and the clique table are applied;
 *  the propagated domains of the variables can be accessed with the usual bound accessing calls SCIPvarGetLbLocal()
 *  and SCIPvarGetUbLocal(); the propagation is only valid locally, i.e. the local bounds as well as the changed
 *  bounds due to SCIPchgVarLbProbing(), SCIPchgVarUbProbing(), and SCIPfixVarProbing() are used for propagation
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPpropagateProbingImplications(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool*            cutoff              /**< pointer to store whether the probing node can be cut off */
   );

/** solves the LP at the current probing node (cannot be applied at preprocessing stage);
 *  no separation or pricing is applied
 *
 *  The LP has to be constructed before (you can use SCIPisLPConstructed() or SCIPconstructLP()).
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsolveProbingLP(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   itlim,              /**< maximal number of LP iterations to perform, or -1 for no limit */
   SCIP_Bool*            lperror,            /**< pointer to store whether an unresolved LP error occurred */
   SCIP_Bool*            cutoff              /**< pointer to store whether the probing LP was infeasible or the objective
                                              *   limit was reached (or NULL, if not needed) */
   );

/** solves the LP at the current probing node (cannot be applied at preprocessing stage) and applies pricing
 *  until the LP is solved to optimality; no separation is applied
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed . See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPsolveProbingLPWithPricing(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             pretendroot,        /**< should the pricers be called as if we are at the root node? */
   SCIP_Bool             displayinfo,        /**< should info lines be displayed after each pricing round? */
   int                   maxpricerounds,     /**< maximal number of pricing rounds (-1: no limit);
                                              *   a finite limit means that the LP might not be solved to optimality! */
   SCIP_Bool*            lperror,            /**< pointer to store whether an unresolved LP error occurred */
   SCIP_Bool*            cutoff              /**< pointer to store whether the probing LP was infeasible or the objective
                                              *   limit was reached (or NULL, if not needed) */

   );

/**@} */




/*
 * branching methods
 */

/**@name Branching Methods */
/**@{ */

/** gets branching candidates for LP solution branching (fractional variables) along with solution values,
 *  fractionalities, and number of branching candidates;
 *  branching rules should always select the branching candidate among the first npriolpcands of the candidate
 *  list
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetLPBranchCands(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           lpcands,            /**< pointer to store the array of LP branching candidates, or NULL */
   SCIP_Real**           lpcandssol,         /**< pointer to store the array of LP candidate solution values, or NULL */
   SCIP_Real**           lpcandsfrac,        /**< pointer to store the array of LP candidate fractionalities, or NULL */
   int*                  nlpcands,           /**< pointer to store the number of LP branching candidates, or NULL */
   int*                  npriolpcands        /**< pointer to store the number of candidates with maximal priority, or NULL */
   );

/** gets number of branching candidates for LP solution branching (number of fractional variables)
 *
 *  @return the number of branching candidates for LP solution branching (number of fractional variables).
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNLPBranchCands(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of branching candidates with maximal priority for LP solution branching
 *
 *  @return the number of branching candidates with maximal priority for LP solution branching.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNPrioLPBranchCands(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets external branching candidates along with solution values, scores, and number of branching candidates;
 *  these branching candidates can be used by relaxations or nonlinear constraint handlers
 *  branching rules should always select the branching candidate among the first nprioexterncands of the candidate
 *  list
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 *
 *  @note Candidate variables with maximal priority are ordered: binaries first, then integers, implicit integers and
 *        continuous last.
 */
EXTERN
SCIP_RETCODE SCIPgetExternBranchCands(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           externcands,        /**< pointer to store the array of extern branching candidates, or NULL */
   SCIP_Real**           externcandssol,     /**< pointer to store the array of extern candidate solution values, or NULL */
   SCIP_Real**           externcandsscore,   /**< pointer to store the array of extern candidate scores, or NULL */
   int*                  nexterncands,       /**< pointer to store the number of extern branching candidates, or NULL */
   int*                  nprioexterncands,   /**< pointer to store the number of candidates with maximal priority, or NULL */
   int*                  nprioexternbins,    /**< pointer to store the number of binary candidates with maximal priority, or NULL */
   int*                  nprioexternints,    /**< pointer to store the number of integer candidates with maximal priority, or NULL */
   int*                  nprioexternimpls    /**< pointer to store the number of implicit integer candidates with maximal priority, 
                                              *   or NULL */
   );

/** gets number of external branching candidates
 *
 *  @return the number of external branching candidates.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNExternBranchCands(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of external branching candidates with maximal branch priority
 *
 *  @return the number of external branching candidates with maximal branch priority.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNPrioExternBranchCands(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of binary external branching candidates with maximal branch priority 
 *
 *  @return the number of binary external branching candidates with maximal branch priority.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNPrioExternBranchBins(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of integer external branching candidates with maximal branch priority
 *
 *  @return the number of integer external branching candidates with maximal branch priority.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNPrioExternBranchInts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of implicit integer external branching candidates with maximal branch priority
 *
 *  @return the number of implicit integer external branching candidates with maximal branch priority.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNPrioExternBranchImpls(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of continuous external branching candidates with maximal branch priority
 *
 *  @return the number of continuous external branching candidates with maximal branch priority.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNPrioExternBranchConts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** insert variable, its score and its solution value into the external branching candidate storage
 * the relative difference of the current lower and upper bounds of a continuous variable must be at least epsilon
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPaddExternBranchCand(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to insert */
   SCIP_Real             score,              /**< score of external candidate, e.g. infeasibility */
   SCIP_Real             solval              /**< value of the variable in the current solution */
   );

/** removes all external candidates from the storage for external branching
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
void SCIPclearExternBranchCands(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** checks whether the given variable is contained in the candidate storage for external branching
 *
 *  @return whether the given variable is contained in the candidate storage for external branching.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Bool SCIPcontainsExternBranchCand(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to look for */
   );

/** gets branching candidates for pseudo solution branching (non-fixed variables) along with the number of candidates
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPgetPseudoBranchCands(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           pseudocands,        /**< pointer to store the array of pseudo branching candidates, or NULL */
   int*                  npseudocands,       /**< pointer to store the number of pseudo branching candidates, or NULL */
   int*                  npriopseudocands    /**< pointer to store the number of candidates with maximal priority, or NULL */
   );

/** gets number of branching candidates for pseudo solution branching (non-fixed variables)
 *
 *  @return the number branching candidates for pseudo solution branching (non-fixed variables).
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNPseudoBranchCands(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of branching candidates with maximal branch priority for pseudo solution branching
 *
 *  @return the number of branching candidates with maximal branch priority for pseudo solution branching.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNPrioPseudoBranchCands(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of binary branching candidates with maximal branch priority for pseudo solution branching
 *
 *  @return the number of binary branching candidates with maximal branch priority for pseudo solution branching.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNPrioPseudoBranchBins(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of integer branching candidates with maximal branch priority for pseudo solution branching
 *
 *  @return the number of integer branching candidates with maximal branch priority for pseudo solution branching.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNPrioPseudoBranchInts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of implicit integer branching candidates with maximal branch priority for pseudo solution branching
 *
 *  @return the number of implicit integer branching candidates with maximal branch priority for pseudo solution branching.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
int SCIPgetNPrioPseudoBranchImpls(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** calculates the branching score out of the gain predictions for a binary branching
 *
 *  @return the branching score out of the gain predictions for a binary branching.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetBranchScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable, of which the branching factor should be applied, or NULL */
   SCIP_Real             downgain,           /**< prediction of objective gain for rounding downwards */
   SCIP_Real             upgain              /**< prediction of objective gain for rounding upwards */
   );

/** calculates the branching score out of the gain predictions for a branching with arbitrary many children
 *
 *  @return the branching score out of the gain predictions for a branching with arbitrary many children.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetBranchScoreMultiple(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable, of which the branching factor should be applied, or NULL */
   int                   nchildren,          /**< number of children that the branching will create */
   SCIP_Real*            gains               /**< prediction of objective gain for each child */
   );

/** computes a branching point for a continuous or discrete variable
 * @see SCIPbranchGetBranchingPoint
 *
 *  @return the branching point for a continuous or discrete variable.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetBranchingPoint(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable, of which the branching point should be computed */
   SCIP_Real             suggestion          /**< suggestion for branching point, or SCIP_INVALID if no suggestion */
   );

/** calculates the node selection priority for moving the given variable's LP value to the given target value;
 *  this node selection priority can be given to the SCIPcreateChild() call
 *
 *  @return the node selection priority for moving the given variable's LP value to the given target value.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPcalcNodeselPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable on which the branching is applied */
   SCIP_BRANCHDIR        branchdir,          /**< type of branching that was performed: upwards, downwards, or fixed;
                                              *   fixed should only be used, when both bounds changed
                                              */
   SCIP_Real             targetvalue         /**< new value of the variable in the child node */
   );

/** calculates an estimate for the objective of the best feasible solution contained in the subtree after applying the given
 *  branching; this estimate can be given to the SCIPcreateChild() call
 *
 *  @return the estimate for the objective of the best feasible solution contained in the subtree after applying the given
 *  branching.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPcalcChildEstimate(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable on which the branching is applied */
   SCIP_Real             targetvalue         /**< new value of the variable in the child node */
   );

/** creates a child node of the focus node
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPcreateChild(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE**           node,               /**< pointer to node data structure */
   SCIP_Real             nodeselprio,        /**< node selection priority of new node */
   SCIP_Real             estimate            /**< estimate for (transformed) objective value of best feasible solution in subtree */
   );

/** branches on a non-continuous variable v using the current LP or pseudo solution;
 *  if solution value x' is fractional, two child nodes will be created
 *  (x <= floor(x'), x >= ceil(x')),
 *  if solution value is integral, the x' is equal to lower or upper bound of the branching
 *  variable and the bounds of v are finite, then two child nodes will be created
 *  (x <= x'', x >= x''+1 with x'' = floor((lb + ub)/2)),
 *  otherwise (up to) three child nodes will be created
 *  (x <= x'-1, x == x', x >= x'+1)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPbranchVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to branch on */
   SCIP_NODE**           downchild,          /**< pointer to return the left child with variable rounded down, or NULL */
   SCIP_NODE**           eqchild,            /**< pointer to return the middle child with variable fixed, or NULL */
   SCIP_NODE**           upchild             /**< pointer to return the right child with variable rounded up, or NULL */
   );

/** branches a variable x using a given domain hole; two child nodes (x <= left, x >= right) are created
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPbranchVarHole(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to branch on */
   SCIP_Real             left,               /**< left side of the domain hole */
   SCIP_Real             right,              /**< right side of the domain hole */
   SCIP_NODE**           downchild,          /**< pointer to return the left child (x <= left), or NULL */
   SCIP_NODE**           upchild             /**< pointer to return the right child (x >= right), or NULL */
   );

/** branches on a variable x using a given value x'; 
 *  for continuous variables with relative domain width larger epsilon, x' must not be one of the bounds;
 *  two child nodes (x <= x', x >= x') are created;
 *  for integer variables, if solution value x' is fractional, two child nodes are created
 *  (x <= floor(x'), x >= ceil(x')),
 *  if x' is integral, three child nodes are created
 *  (x <= x'-1, x == x', x >= x'+1)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPbranchVarVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to branch on */
   SCIP_Real             val,                /**< value to branch on */
   SCIP_NODE**           downchild,          /**< pointer to return the left child with variable rounded down, or NULL */
   SCIP_NODE**           eqchild,            /**< pointer to return the middle child with variable fixed, or NULL */
   SCIP_NODE**           upchild             /**< pointer to return the right child with variable rounded up, or NULL */
   );

/** n-ary branching on a variable x using a given value
 * Branches on variable x such that up to n/2 children are created on each side of the usual branching value.
 * The branching value is selected as in SCIPbranchVarVal().
 * The parameters minwidth and widthfactor determine the domain width of the branching variable in the child nodes.
 * If n is odd, one child with domain width 'width' and having the branching value in the middle is created.
 * Otherwise, two children with domain width 'width' and being left and right of the branching value are created.
 * Next further nodes to the left and right are created, where width is multiplied by widthfactor with increasing distance from the first nodes.
 * The initial width is calculated such that n/2 nodes are created to the left and to the right of the branching value.
 * If this value is below minwidth, the initial width is set to minwidth, which may result in creating less than n nodes.
 *
 * Giving a large value for widthfactor results in creating children with small domain when close to the branching value
 * and large domain when closer to the current variable bounds. That is, setting widthfactor to a very large value and n to 3
 * results in a ternary branching where the branching variable is mostly fixed in the middle child.
 * Setting widthfactor to 1.0 results in children where the branching variable always has the same domain width
 * (except for one child if the branching value is not in the middle).
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPbranchVarValNary(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to branch on */
   SCIP_Real             val,                /**< value to branch on */
   int                   n,                  /**< attempted number of children to be created, must be >= 2 */
   SCIP_Real             minwidth,           /**< minimal domain width in children */
   SCIP_Real             widthfactor,        /**< multiplier for children domain width with increasing distance from val, must be >= 1.0 */
   int*                  nchildren           /**< pointer to store number of created children, or NULL */
   );

/** calls branching rules to branch on an LP solution; if no fractional variables exist, the result is SCIP_DIDNOTRUN;
 *  if the branch priority of an unfixed variable is larger than the maximal branch priority of the fractional
 *  variables, pseudo solution branching is applied on the unfixed variables with maximal branch priority
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPbranchLP(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RESULT*          result              /**< pointer to store the result of the branching (s. branch.h) */
   );

/** calls branching rules to branch on a external candidates; if no such candidates exist, the result is SCIP_DIDNOTRUN
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPbranchExtern(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RESULT*          result              /**< pointer to store the result of the branching (s. branch.h) */
   );

/** calls branching rules to branch on a pseudo solution; if no unfixed variables exist, the result is SCIP_DIDNOTRUN
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPbranchPseudo(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RESULT*          result              /**< pointer to store the result of the branching (s. branch.h) */
   );

/**@} */




/*
 * primal solutions
 */

/**@name Primal Solution Methods */
/**@{ */

/** creates a primal solution, initialized to zero
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   );

/** creates a primal solution, initialized to the current LP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateLPSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   );

/** creates a primal solution, initialized to the current NLP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateNLPSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   );

/** creates a primal solution, initialized to the current relaxation solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateRelaxSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   );

/** creates a primal solution, initialized to the current pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreatePseudoSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   );

/** creates a primal solution, initialized to the current LP or pseudo solution, depending on whether the LP was solved
 *  at the current node
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateCurrentSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   );

/** creates a primal solution, initialized to unknown values
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateUnknownSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   );

/** creates a primal solution living in the original problem space, initialized to zero;
 *  a solution in original space allows to set original variables to values that would be invalid in the
 *  transformed problem due to preprocessing fixings or aggregations
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateOrigSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   );

/** creates a copy of a primal solution; note that a copy of a linked solution is also linked and needs to be unlinked
 *  if it should stay unaffected from changes in the LP or pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcreateSolCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_SOL*             sourcesol           /**< primal CIP solution to copy */
   );

/** creates a copy of a primal solution, thereby replacing infinite fixings of variables by finite values;
 *  the copy is always defined in the original variable space
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPcreateSolCopyRemoveInfiniteFixings(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_SOL*             sourcesol,          /**< primal CIP solution to copy */
   SCIP_Bool*            success             /**< could infinite fixings be removed? */
   );

/** frees primal CIP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPfreeSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol                 /**< pointer to the solution */
   );

/** links a primal solution to the current LP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPlinkLPSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   );

/** links a primal solution to the current NLP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPlinkNLPSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   );

/** links a primal solution to the current relaxation solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPlinkRelaxSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   );

/** links a primal solution to the current pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPlinkPseudoSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   );

/** links a primal solution to the current LP or pseudo solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPlinkCurrentSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   );

/** clears a primal solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPclearSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   );

/** stores solution values of variables in solution's own array
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPunlinkSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   );

/** sets value of variable in primal CIP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPsetSolVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution */
   SCIP_VAR*             var,                /**< variable to add to solution */
   SCIP_Real             val                 /**< solution value of variable */
   );

/** sets values of multiple variables in primal CIP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPsetSolVals(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution */
   int                   nvars,              /**< number of variables to set solution value for */
   SCIP_VAR**            vars,               /**< array with variables to add to solution */
   SCIP_Real*            vals                /**< array with solution values of variables */
   );

/** increases value of variable in primal CIP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPincSolVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution */
   SCIP_VAR*             var,                /**< variable to increase solution value for */
   SCIP_Real             incval              /**< increment for solution value of variable */
   );

/** returns value of variable in primal CIP solution, or in current LP/pseudo solution
 *
 *  @return value of variable in primal CIP solution, or in current LP/pseudo solution
 *
 *  @pre In case the solution pointer @p sol is @b NULL, that means it is asked for the LP or pseudo solution, this method
 *       can only be called if @p scip is in the solving stage \ref SCIP_STAGE_SOLVING. In any other case, this method
 *       can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Real SCIPgetSolVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution, or NULL for current LP/pseudo solution */
   SCIP_VAR*             var                 /**< variable to get value for */
   );

/** gets values of multiple variables in primal CIP solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPgetSolVals(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution, or NULL for current LP/pseudo solution */
   int                   nvars,              /**< number of variables to get solution value for */
   SCIP_VAR**            vars,               /**< array with variables to get value for */
   SCIP_Real*            vals                /**< array to store solution values of variables */
   );

/** returns objective value of primal CIP solution w.r.t. original problem, or current LP/pseudo objective value
 *
 *  @return objective value of primal CIP solution w.r.t. original problem, or current LP/pseudo objective value
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Real SCIPgetSolOrigObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution, or NULL for current LP/pseudo objective value */
   );

/** returns transformed objective value of primal CIP solution, or transformed current LP/pseudo objective value
 *
 *  @return transformed objective value of primal CIP solution, or transformed current LP/pseudo objective value
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Real SCIPgetSolTransObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution, or NULL for current LP/pseudo objective value */
   );

/** maps original space objective value into transformed objective value
 *
 *  @return transformed objective value
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPtransformObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             obj                 /**< original space objective value to transform */
   );

/** maps transformed objective value into original space
 *
 *  @return objective value into original space
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPretransformObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             obj                 /**< transformed objective value to retransform in original space */
   );

/** gets clock time, when this solution was found
 *
 *  @return clock time, when this solution was found
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Real SCIPgetSolTime(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   );

/** gets branch and bound run number, where this solution was found
 *
 *  @return branch and bound run number, where this solution was found
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
int SCIPgetSolRunnum(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   );

/** gets node number of the specific branch and bound run, where this solution was found
 *
 *  @return node number of the specific branch and bound run, where this solution was found
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Longint SCIPgetSolNodenum(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   );

/** gets heuristic, that found this solution (or NULL if it's from the tree)
 *
 *  @return heuristic, that found this solution (or NULL if it's from the tree)
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_HEUR* SCIPgetSolHeur(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   );

/** returns whether two given solutions are exactly equal
 *
 *  @return returns whether two given solutions are exactly equal
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Bool SCIPareSolsEqual(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol1,               /**< first primal CIP solution */
   SCIP_SOL*             sol2                /**< second primal CIP solution */
   );

/** outputs non-zero variables of solution in original problem space to the given file stream
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre In case the solution pointer @p sol is NULL (askinking for the current LP/pseudo solution), this method can be
 *       called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *
 *  @pre In case the solution pointer @p sol is @b not NULL, this method can be called if @p scip is in one of the
 *       following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPprintSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution, or NULL for current LP/pseudo solution */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_Bool             printzeros          /**< should variables set to zero be printed? */
   );

/** outputs non-zero variables of solution in transformed problem space to file stream
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPprintTransSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution, or NULL for current LP/pseudo solution */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_Bool             printzeros          /**< should variables set to zero be printed? */
   );

/** outputs non-zero variables of solution representing a ray in original problem space to file stream
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPprintRay(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution representing ray */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_Bool             printzeros          /**< should variables set to zero be printed? */
   );

/** gets number of feasible primal solutions stored in the solution storage in case the problem is transformed; in case
 *  if the problem stage is SCIP_STAGE_PROBLEM, it returns the number solution in the original solution candidate
 *  storage
 *
 *  @return number of feasible primal solutions stored in the solution storage in case the problem is transformed; in case
 *          if the problem stage is SCIP_STAGE_PROBLEM, it returns the number solution in the original solution candidate
 *          storage
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetNSols(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets array of feasible primal solutions stored in the solution storage in case the problem is transformed; in case
 *  if the problem stage is in SCIP_STAGE_PROBLEM, it returns the number array of solution candidate stored
 *
 *  @return array of feasible primal solutions
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_SOL** SCIPgetSols(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets best feasible primal solution found so far if the problem is transformed; in case the problem is in
 *  SCIP_STAGE_PROBLEM it returns the best solution candidate, or NULL if no solution has been found or the candidate
 *  store is empty;
 *
 *  @return best feasible primal solution so far
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_SOL* SCIPgetBestSol(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** outputs best feasible primal solution found so far to file stream
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREE
 */
EXTERN
SCIP_RETCODE SCIPprintBestSol(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_Bool             printzeros          /**< should variables set to zero be printed? */
   );

/** outputs best feasible primal solution found so far in transformed variables to file stream
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREE
 */
EXTERN
SCIP_RETCODE SCIPprintBestTransSol(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_Bool             printzeros          /**< should variables set to zero be printed? */
   );

/** try to round given solution
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIProundSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution */
   SCIP_Bool*            success             /**< pointer to store whether rounding was successful */
   );

/** retransforms solution to original problem space
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPretransformSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal CIP solution */
   );

/** reads a given solution file, problem has to be transformed in advance
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPreadSol(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename            /**< name of the input file */
   );

/** adds feasible primal solution to solution storage by copying it
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPaddSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution */
   SCIP_Bool*            stored              /**< stores whether given solution was good enough to keep */
   );

/** adds primal solution to solution storage, frees the solution afterwards
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPaddSolFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to primal CIP solution; is cleared in function call */
   SCIP_Bool*            stored              /**< stores whether given solution was good enough to keep */
   );

/** adds current LP/pseudo solution to solution storage
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPaddCurrentSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< heuristic that found the solution */
   SCIP_Bool*            stored              /**< stores whether given solution was good enough to keep */
   );

/** checks solution for feasibility; if possible, adds it to storage by copying
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPtrySol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution */
   SCIP_Bool             printreason,        /**< should all reasons of violations be printed? */
   SCIP_Bool             checkbounds,        /**< should the bounds of the variables be checked? */
   SCIP_Bool             checkintegrality,   /**< has integrality to be checked? */
   SCIP_Bool             checklprows,        /**< have current LP rows (both local and global) to be checked? */
   SCIP_Bool*            stored              /**< stores whether given solution was feasible and good enough to keep */
   );

/** checks primal solution; if feasible, adds it to storage; solution is freed afterwards
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPtrySolFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to primal CIP solution; is cleared in function call */
   SCIP_Bool             printreason,        /**< should all reasons of violations be printed? */
   SCIP_Bool             checkbounds,        /**< should the bounds of the variables be checked? */
   SCIP_Bool             checkintegrality,   /**< has integrality to be checked? */
   SCIP_Bool             checklprows,        /**< have current LP rows (both local and global) to be checked? */
   SCIP_Bool*            stored              /**< stores whether solution was feasible and good enough to keep */
   );

/** checks current LP/pseudo solution for feasibility; if possible, adds it to storage
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPtryCurrentSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< heuristic that found the solution */
   SCIP_Bool             printreason,        /**< should all reasons of violations be printed? */
   SCIP_Bool             checkintegrality,   /**< has integrality to be checked? */
   SCIP_Bool             checklprows,        /**< have current LP rows (both local and global) to be checked? */
   SCIP_Bool*            stored              /**< stores whether given solution was feasible and good enough to keep */
   );

/** checks solution for feasibility without adding it to the solution store
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPcheckSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution */
   SCIP_Bool             printreason,        /**< should all reasons of violations be printed? */
   SCIP_Bool             checkbounds,        /**< should the bounds of the variables be checked? */
   SCIP_Bool             checkintegrality,   /**< has integrality to be checked? */
   SCIP_Bool             checklprows,        /**< have current LP rows (both local and global) to be checked? */
   SCIP_Bool*            feasible            /**< stores whether given solution is feasible */
   );

/** checks solution for feasibility in original problem without adding it to the solution store;
 *  this method is used to double check a solution in order to validate the presolving process
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPcheckSolOrig(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution */
   SCIP_Bool*            feasible,           /**< stores whether given solution is feasible */
   SCIP_Bool             printreason,        /**< should the reason for the violation be printed? */
   SCIP_Bool             completely          /**< should all violations be checked? */
   );

/** return whether a primal ray is stored that proves unboundedness of the LP relaxation
 *
 *  @return return whether a primal ray is stored that proves unboundedness of the LP relaxation
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Bool SCIPhasPrimalRay(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets value of given variable in primal ray causing unboundedness of the LP relaxation;
 *  should only be called if such a ray is stored (check with SCIPhasPrimalRay())
 *
 *  @return value of given variable in primal ray causing unboundedness of the LP relaxation
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetPrimalRayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get value for */
   );

/**@} */




/*
 * event methods
 */

/**@name Event Methods
 *
 * Events can only be caught during the operation on the transformed problem.
 * Events on variables can only be caught for transformed variables.
 * If you want to catch an event for an original variable, you have to get the corresponding transformed variable
 * with a call to SCIPgetTransformedVar() and catch the event on the transformed variable.
 */
/**@{ */

/** catches a global (not variable or row dependent) event
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPcatchEvent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EVENTTYPE        eventtype,          /**< event type mask to select events to catch */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler to process events with */
   SCIP_EVENTDATA*       eventdata,          /**< event data to pass to the event handler when processing this event */
   int*                  filterpos           /**< pointer to store position of event filter entry, or NULL */
   );

/** drops a global event (stops to track event)
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPdropEvent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EVENTTYPE        eventtype,          /**< event type mask of dropped event */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler to process events with */
   SCIP_EVENTDATA*       eventdata,          /**< event data to pass to the event handler when processing this event */
   int                   filterpos           /**< position of event filter entry returned by SCIPcatchEvent(), or -1 */
   );

/** catches an objective value or domain change event on the given transformed variable
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPcatchVarEvent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< transformed variable to catch event for */
   SCIP_EVENTTYPE        eventtype,          /**< event type mask to select events to catch */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler to process events with */
   SCIP_EVENTDATA*       eventdata,          /**< event data to pass to the event handler when processing this event */
   int*                  filterpos           /**< pointer to store position of event filter entry, or NULL */
   );

/** drops an objective value or domain change event (stops to track event) on the given transformed variable
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPdropVarEvent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< transformed variable to drop event for */
   SCIP_EVENTTYPE        eventtype,          /**< event type mask of dropped event */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler to process events with */
   SCIP_EVENTDATA*       eventdata,          /**< event data to pass to the event handler when processing this event */
   int                   filterpos           /**< position of event filter entry returned by SCIPcatchVarEvent(), or -1 */
   );

/** catches a row coefficient, constant, or side change event on the given row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPcatchRowEvent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< linear row to catch event for */
   SCIP_EVENTTYPE        eventtype,          /**< event type mask to select events to catch */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler to process events with */
   SCIP_EVENTDATA*       eventdata,          /**< event data to pass to the event handler when processing this event */
   int*                  filterpos           /**< pointer to store position of event filter entry, or NULL */
   );

/** drops a row coefficient, constant, or side change event (stops to track event) on the given row
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPdropRowEvent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< linear row to drop event for */
   SCIP_EVENTTYPE        eventtype,          /**< event type mask of dropped event */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler to process events with */
   SCIP_EVENTDATA*       eventdata,          /**< event data to pass to the event handler when processing this event */
   int                   filterpos           /**< position of event filter entry returned by SCIPcatchVarEvent(), or -1 */
   );

/**@} */




/*
 * tree methods
 */

/**@name Tree Methods */
/**@{ */

/** gets current node in the tree
 *
 *  @return the current node of the search tree
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NODE* SCIPgetCurrentNode(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the root node of the tree
 *
 *  @return the root node of the search tree
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NODE* SCIPgetRootNode(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns whether the current node is already solved and only propagated again
 *
 *  @return TRUE is returned if \SCIP performance repropagation, otherwise FALSE.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_Bool SCIPinRepropagation(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets children of focus node along with the number of children
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetChildren(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE***          children,           /**< pointer to store children array, or NULL if not needed */
   int*                  nchildren           /**< pointer to store number of children, or NULL if not needed */
   );

/** gets number of children of focus node
 *
 *  @return number of children of the focus node
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetNChildren(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets siblings of focus node along with the number of siblings
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetSiblings(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE***          siblings,           /**< pointer to store siblings array, or NULL if not needed */
   int*                  nsiblings           /**< pointer to store number of siblings, or NULL if not needed */
   );

/** gets number of siblings of focus node
 *
 *  @return the number of siblings of focus node
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetNSiblings(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets leaves of the tree along with the number of leaves
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPgetLeaves(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE***          leaves,             /**< pointer to store leaves array, or NULL if not needed */
   int*                  nleaves             /**< pointer to store number of leaves, or NULL if not needed */
   );

/** gets number of leaves in the tree
 *
 *  @return the number of leaves in the tree
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetNLeaves(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the best child of the focus node w.r.t. the node selection priority assigned by the branching rule
 *
 *  @return the best child of the focus node w.r.t. the node selection priority assigned by the branching rule
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NODE* SCIPgetPrioChild(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the best sibling of the focus node w.r.t. the node selection priority assigned by the branching rule
 *
 *  @return the best sibling of the focus node w.r.t. the node selection priority assigned by the branching rule
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NODE* SCIPgetPrioSibling(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the best child of the focus node w.r.t. the node selection strategy
 *
 *  @return the best child of the focus node w.r.t. the node selection strategy
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NODE* SCIPgetBestChild(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the best sibling of the focus node w.r.t. the node selection strategy
 *
 *  @return the best sibling of the focus node w.r.t. the node selection strategy
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NODE* SCIPgetBestSibling(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the best leaf from the node queue w.r.t. the node selection strategy
 *
 *  @return the best leaf from the node queue w.r.t. the node selection strategy
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NODE* SCIPgetBestLeaf(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the best node from the tree (child, sibling, or leaf) w.r.t. the node selection strategy
 *
 *  @return the best node from the tree (child, sibling, or leaf) w.r.t. the node selection strategy
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NODE* SCIPgetBestNode(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the node with smallest lower bound from the tree (child, sibling, or leaf)
 *
 *  @return the node with smallest lower bound from the tree (child, sibling, or leaf)
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_NODE* SCIPgetBestboundNode(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** cuts off node and whole sub tree from branch and bound tree
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPcutoffNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node                /**< node that should be cut off */
   );

/** marks the given node to be propagated again the next time a node of its subtree is processed
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPrepropagateNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node                /**< node that should be propagated again */
   );

/** returns depth of first node in active path that is marked being cutoff
 *
 *  @return depth of first node in active path that is marked being cutoff
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetCutoffdepth(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns depth of first node in active path that has to be propagated again
 *
 *  @return depth of first node in active path that has to be propagated again
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetRepropdepth(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** prints all branching decisions on variables from the root to the given node
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if @p scip is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPprintNodeRootPath(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node data */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/**@} */


/*
 * statistic methods
 */

/**@name Statistic Methods */
/**@{ */

/** gets number of branch and bound runs performed, including the current run
 *
 *  @return the number of branch and bound runs performed, including the current run
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
int SCIPgetNRuns(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of processed nodes in current run, including the focus node
 *
 *  @return the number of processed nodes in current run, including the focus node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Longint SCIPgetNNodes(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of processed nodes in all runs, including the focus node
 *
 *  @return the total number of processed nodes in all runs, including the focus node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_Longint SCIPgetNTotalNodes(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of nodes left in the tree (children + siblings + leaves)
 *
 *  @return the number of nodes left in the tree (children + siblings + leaves)
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNNodesLeft(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of LPs solved so far
 *
 *  @return the total number of LPs solved so far
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNLPs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of iterations used so far in primal and dual simplex and barrier algorithm
 *
 *  @return the total number of iterations used so far in primal and dual simplex and barrier algorithm
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of iterations used so far in primal and dual simplex and barrier algorithm for the root node
 *
 *  @return the total number of iterations used so far in primal and dual simplex and barrier algorithm for the root node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNRootLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of iterations used in primal and dual simplex and barrier algorithm for the first LP at the root
 *  node
 *
 *  @return the total number of iterations used in primal and dual simplex and barrier algorithm for the first root LP
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNRootFirstLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of primal LPs solved so far
 *
 *  @return the total number of primal LPs solved so far
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNPrimalLPs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of iterations used so far in primal simplex
 *
 *  @return total number of iterations used so far in primal simplex
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNPrimalLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of dual LPs solved so far
 *
 *  @return the total number of dual LPs solved so far
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNDualLPs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of iterations used so far in dual simplex
 *
 *  @return the total number of iterations used so far in dual simplex
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNDualLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of barrier LPs solved so far
 *
 *  @return the total number of barrier LPs solved so far
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNBarrierLPs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of iterations used so far in barrier algorithm
 *
 *  @return the total number of iterations used so far in barrier algorithm
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNBarrierLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of LPs solved so far that were resolved from an advanced start basis
 *
 *  @return the total number of LPs solved so far that were resolved from an advanced start basis
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNResolveLPs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of simplex iterations used so far in primal and dual simplex calls where an advanced start basis
 *  was available
 *
 *  @return the total number of simplex iterations used so far in primal and dual simplex calls where an advanced start
 *          basis was available
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNResolveLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of primal LPs solved so far that were resolved from an advanced start basis
 *
 *  @return the total number of primal LPs solved so far that were resolved from an advanced start basis
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNPrimalResolveLPs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of simplex iterations used so far in primal simplex calls where an advanced start basis
 *  was available
 *
 *  @return the total number of simplex iterations used so far in primal simplex calls where an advanced start
 *          basis was available
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNPrimalResolveLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of dual LPs solved so far that were resolved from an advanced start basis
 *
 *  @return the total number of dual LPs solved so far that were resolved from an advanced start basis
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNDualResolveLPs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of simplex iterations used so far in dual simplex calls where an advanced start basis
 *  was available
 *
 *  @return the total number of simplex iterations used so far in dual simplex calls where an advanced start
 *          basis was available
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNDualResolveLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of LPs solved so far for node relaxations
 *
 *  @return the total number of LPs solved so far for node relaxations
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNNodeLPs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of simplex iterations used so far for node relaxations
 *
 *  @return the total number of simplex iterations used so far for node relaxations
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNNodeLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of LPs solved so far for initial LP in node relaxations
 *
 *  @return the total number of LPs solved so far for initial LP in node relaxations
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNNodeInitLPs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of simplex iterations used so far for initial LP in node relaxations
 *
 *  @return the total number of simplex iterations used so far for initial LP in node relaxations
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNNodeInitLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of LPs solved so far during diving and probing
 *
 *  @return total number of LPs solved so far during diving and probing
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNDivingLPs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of simplex iterations used so far during diving and probing
 *
 *  @return the total number of simplex iterations used so far during diving and probing
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNDivingLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of times, strong branching was called (each call represents solving two LPs)
 *
 *  @return the total number of times, strong branching was called (each call represents solving two LPs)
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNStrongbranchs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of simplex iterations used so far in strong branching
 *
 *  @return the total number of simplex iterations used so far in strong branching
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNStrongbranchLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of times, strong branching was called at the root node (each call represents solving two LPs)
 *
 *  @return the total number of times, strong branching was called at the root node (each call represents solving two LPs)
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNRootStrongbranchs(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of simplex iterations used so far in strong branching at the root node
 *
 *  @return the total number of simplex iterations used so far in strong branching at the root node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Longint SCIPgetNRootStrongbranchLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of pricing rounds performed so far at the current node
 *
 *  @return the number of pricing rounds performed so far at the current node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetNPriceRounds(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** get current number of variables in the pricing store
 *
 *  @return the current number of variables in the pricing store
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNPricevars(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** get total number of pricing variables found so far
 *
 *  @return the total number of pricing variables found so far
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNPricevarsFound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** get total number of pricing variables applied to the LPs
 *
 *  @return the total number of pricing variables applied to the LPs
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNPricevarsApplied(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of separation rounds performed so far at the current node
 *
 *  @return the number of separation rounds performed so far at the current node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetNSepaRounds(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** get total number of cuts found so far
 *
 *  @return the total number of cuts found so far
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNCutsFound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** get number of cuts found so far in current separation round
 *
 *  @return the number of cuts found so far in current separation round
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNCutsFoundRound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** get total number of cuts applied to the LPs
 *
 *  @return the total number of cuts applied to the LPs
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNCutsApplied(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** get total number of constraints found in conflict analysis (conflict and reconvergence constraints)
 *
 *  @return the total number of constraints found in conflict analysis (conflict and reconvergence constraints)
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Longint SCIPgetNConflictConssFound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** get number of conflict constraints found so far at the current node
 *
 *  @return the number of conflict constraints found so far at the current node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetNConflictConssFoundNode(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** get total number of conflict constraints added to the problem
 *
 *  @return the total number of conflict constraints added to the problem
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Longint SCIPgetNConflictConssApplied(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets depth of current node, or -1 if no current node exists; in probing, the current node is the last probing node,
 *  such that the depth includes the probing path
 *
 *  @return the depth of current node, or -1 if no current node exists; in probing, the current node is the last probing node,
 *  such that the depth includes the probing path
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetDepth(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets depth of the focus node, or -1 if no focus node exists; the focus node is the currently processed node in the
 *  branching tree, excluding the nodes of the probing path
 *
 *  @return the depth of the focus node, or -1 if no focus node exists; the focus node is the currently processed node in the
 *  branching tree, excluding the nodes of the probing path
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetFocusDepth(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets maximal depth of all processed nodes in current branch and bound run (excluding probing nodes)
 *
 *  @return the maximal depth of all processed nodes in current branch and bound run (excluding probing nodes)
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetMaxDepth(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets maximal depth of all processed nodes over all branch and bound runs
 *
 *  @return the maximal depth of all processed nodes over all branch and bound runs
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
int SCIPgetMaxTotalDepth(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of backtracks, i.e. number of times, the new node was selected from the leaves queue
 *
 *  @return the total number of backtracks, i.e. number of times, the new node was selected from the leaves queue
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Longint SCIPgetNBacktracks(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current plunging depth (successive times, a child was selected as next node)
 *
 *  @return the current plunging depth (successive times, a child was selected as next node)
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetPlungeDepth(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of active constraints at the current node
 *
 *  @return the total number of active constraints at the current node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetNActiveConss(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets total number of enabled constraints at the current node
 *
 *  @return the total number of enabled constraints at the current node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
int SCIPgetNEnabledConss(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets average dual bound of all unprocessed nodes for original problem */
EXTERN
SCIP_Real SCIPgetAvgDualbound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets average lower (dual) bound of all unprocessed nodes in transformed problem
 *
 *  @return the average lower (dual) bound of all unprocessed nodes in transformed problem
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgLowerbound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets global dual bound
 *
 *  @return the global dual bound
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetDualbound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets global lower (dual) bound in transformed problem
 *
 *  @return the global lower (dual) bound in transformed problem
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetLowerbound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets dual bound of the root node for the original problem
 *
 *  @return the dual bound of the root node for the original problem
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetDualboundRoot(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets lower (dual) bound in transformed problem of the root node
 *
 *  @return the lower (dual) bound in transformed problem of the root node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetLowerboundRoot(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets dual bound for the original problem obtained by the first LP solve at the root node
 *
 *  @return the dual bound for the original problem of the first LP solve at the root node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetFirstLPDualboundRoot(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets lower (dual) bound in transformed problem obtained by the first LP solve at the root node
 *
 *  @return the lower (dual) bound in transformed problem obtained by first LP solve at the root node
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetFirstLPLowerboundRoot(
   SCIP*                 scip                /**< SCIP data structure */
   );


/** gets global primal bound (objective value of best solution or user objective limit) for the original problem
 *
 *  @return the global primal bound (objective value of best solution or user objective limit) for the original problem
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Real SCIPgetPrimalbound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets global upper (primal) bound in transformed problem (objective value of best solution or user objective limit)
 *
 *  @return the global upper (primal) bound in transformed problem (objective value of best solution or user objective limit)
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Real SCIPgetUpperbound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets global cutoff bound in transformed problem: a sub problem with lower bound larger than the cutoff
 *  cannot contain a better feasible solution; usually, this bound is equal to the upper bound, but if the
 *  objective value is always integral, the cutoff bound is (nearly) one less than the upper bound;
 *  additionally, due to objective function domain propagation, the cutoff bound can be further reduced
 *
 *  @return global cutoff bound in transformed problem
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Real SCIPgetCutoffbound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** updates the cutoff bound
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *
 *  @note the given cutoff bound has to better or equal to known one (SCIPgetCutoffbound())
 */
EXTERN
SCIP_RETCODE SCIPupdateCutoffbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             cutoffbound         /**< new cutoff bound */
   );

/** returns whether the current primal bound is justified with a feasible primal solution; if not, the primal bound
 *  was set from the user as objective limit
 *
 *  @return TRUE if the current primal bound is justified with a feasible primal solution, otherwise FALSE
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Bool SCIPisPrimalboundSol(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current gap |(primalbound - dualbound)/min(|primalbound|,|dualbound|)| if both bounds have same sign,
 *  or infinity, if they have opposite sign
 *
 *  @return the current gap |(primalbound - dualbound)/min(|primalbound|,|dualbound|)| if both bounds have same sign,
 *  or infinity, if they have opposite sign
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetGap(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets current gap |(upperbound - lowerbound)/min(|upperbound|,|lowerbound|)| in transformed problem if both bounds
 *  have same sign, or infinity, if they have opposite sign
 *
 *  @return current gap |(upperbound - lowerbound)/min(|upperbound|,|lowerbound|)| in transformed problem if both bounds
 *  have same sign, or infinity, if they have opposite sign
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetTransGap(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of feasible primal solutions found so far
 *
 *  @return the number of feasible primal solutions found so far
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Longint SCIPgetNSolsFound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of feasible primal solutions respecting the objective limit found so far
 *
 *  @return the number of feasible primal solutions respecting the objective limit found so far
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Longint SCIPgetNLimSolsFound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets number of feasible primal solutions found so far, that improved the primal bound at the time they were found
 *
 *  @return the number of feasible primal solutions found so far, that improved the primal bound at the time they were found
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_Longint SCIPgetNBestSolsFound(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the average pseudo cost value for the given direction over all variables
 *
 *  @return the average pseudo cost value for the given direction over all variables
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgPseudocost(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             solvaldelta         /**< difference of variable's new LP value - old LP value */
   );

/** gets the average pseudo cost value for the given direction over all variables,
 *  only using the pseudo cost information of the current run
 *
 *  @return the average pseudo cost value for the given direction over all variables,
 *  only using the pseudo cost information of the current run
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgPseudocostCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             solvaldelta         /**< difference of variable's new LP value - old LP value */
   );

/** gets the average number of pseudo cost updates for the given direction over all variables
 *
 *  @return the average number of pseudo cost updates for the given direction over all variables
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgPseudocostCount(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** gets the average number of pseudo cost updates for the given direction over all variables,
 *  only using the pseudo cost information of the current run
 *
 *  @return the average number of pseudo cost updates for the given direction over all variables,
 *  only using the pseudo cost information of the current run
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgPseudocostCountCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** gets the average pseudo cost score value over all variables, assuming a fractionality of 0.5
 *
 *  @return the average pseudo cost score value over all variables, assuming a fractionality of 0.5
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgPseudocostScore(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the average pseudo cost score value over all variables, assuming a fractionality of 0.5,
 *  only using the pseudo cost information of the current run
 *
 *  @return the average pseudo cost score value over all variables, assuming a fractionality of 0.5,
 *  only using the pseudo cost information of the current run
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgPseudocostScoreCurrentRun(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the average conflict score value over all variables */
EXTERN
SCIP_Real SCIPgetAvgConflictScore(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the average conflict score value over all variables, only using the pseudo cost information of the current run
 *
 *  @return the average conflict score value over all variables, only using the pseudo cost information of the current run
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgConflictScoreCurrentRun(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the average inference score value over all variables
 *
 *  @return the average inference score value over all variables
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgConflictlengthScore(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the average conflictlength score value over all variables, only using the pseudo cost information of the
 *  current run
 *
 *  @return the average conflictlength score value over all variables, only using the pseudo cost information of the
 *          current run
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgConflictlengthScoreCurrentRun(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the average number of inferences found after branching in given direction over all variables
 *
 *  @return the average number of inferences found after branching in given direction over all variables
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgInferences(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the average number of inferences found after branching in given direction over all variables,
 *  only using the pseudo cost information of the current run
 *
 *  @return the average number of inferences found after branching in given direction over all variables,
 *          only using the pseudo cost information of the current run
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgInferencesCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** gets the average inference score value over all variables
 *
 *  @return the average inference score value over all variables
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgInferenceScore(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the average inference score value over all variables, only using the inference information information of the
 *  current run
 *
 *  @return the average inference score value over all variables, only using the inference information information of the
 *          current run
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgInferenceScoreCurrentRun(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the average number of cutoffs found after branching in given direction over all variables
 *
 *  @return the average number of cutoffs found after branching in given direction over all variables
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgCutoffs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the average number of cutoffs found after branching in given direction over all variables,
 *  only using the pseudo cost information of the current run
 *
 *  @return the average number of cutoffs found after branching in given direction over all variables,
 *          only using the pseudo cost information of the current run
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgCutoffsCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** gets the average cutoff score value over all variables
 *
 *  @return the average cutoff score value over all variables
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgCutoffScore(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the average cutoff score value over all variables, only using the pseudo cost information of the current run
 *
 *  @return the average cutoff score value over all variables, only using the pseudo cost information of the current run
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_Real SCIPgetAvgCutoffScoreCurrentRun(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** outputs original problem to file stream
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPprintOrigProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   const char*           extension,          /**< file format (or NULL for default CIP format)*/
   SCIP_Bool             genericnames        /**< using generic variable and constraint names? */
   );

/** outputs transformed problem of the current node to file stream
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 */
EXTERN
SCIP_RETCODE SCIPprintTransProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   const char*           extension,          /**< file format (or NULL for default CIP format)*/
   SCIP_Bool             genericnames        /**< using generic variable and constraint names? */
   );

/** outputs solving statistics
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPprintStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/** outputs history statistics about branchings on variables
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INIT
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
SCIP_RETCODE SCIPprintBranchingStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   );

/** outputs node information display line
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_SOLVING
 */
EXTERN
SCIP_RETCODE SCIPprintDisplayLine(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_VERBLEVEL        verblevel,          /**< minimal verbosity level to actually display the information line */
   SCIP_Bool             endline             /**< should the line be terminated with a newline symbol? */
   );

/** gets total number of implications between variables that are stored in the implication graph
 *
 *  @return the total number of implications between variables that are stored in the implication graph
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 */
EXTERN
int SCIPgetNImplications(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** stores conflict graph of binary variables' implications into a file, which can be used as input for the DOT tool
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 */
EXTERN
SCIP_RETCODE SCIPwriteImplicationConflictGraph(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename            /**< file name, or NULL for stdout */
   );

/**@} */




/*
 * timing methods
 */

/**@name Timing Methods */
/**@{ */

/** gets current time of day in seconds (standard time zone)
 *
 *  @return the current time of day in seconds (standard time zone).
 */
EXTERN
SCIP_Real SCIPgetTimeOfDay(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** creates a clock using the default clock type
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPcreateClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK**          clck                /**< pointer to clock timer */
   );

/** creates a clock counting the CPU user seconds
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPcreateCPUClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK**          clck                /**< pointer to clock timer */
   );

/** creates a clock counting the wall clock seconds
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPcreateWallClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK**          clck                /**< pointer to clock timer */
   );

/** frees a clock
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPfreeClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK**          clck                /**< pointer to clock timer */
   );

/** resets the time measurement of a clock to zero and completely stops the clock
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPresetClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK*           clck                /**< clock timer */
   );

/** starts the time measurement of a clock
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPstartClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK*           clck                /**< clock timer */
   );

/** stops the time measurement of a clock
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPstopClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK*           clck                /**< clock timer */
   );

/** starts the current solving time
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPstartSolvingTime(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** stops the current solving time in seconds
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *       - \ref SCIP_STAGE_EXITSOLVE
 *       - \ref SCIP_STAGE_FREETRANS
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_RETCODE SCIPstopSolvingTime(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the measured time of a clock in seconds
 *
 *  @return the measured time of a clock in seconds.
 */
EXTERN
SCIP_Real SCIPgetClockTime(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK*           clck                /**< clock timer */
   );

/** sets the measured time of a clock to the given value in seconds
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetClockTime(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK*           clck,               /**< clock timer */
   SCIP_Real             sec                 /**< time in seconds to set the clock's timer to */
   );

/** gets the current total SCIP time in seconds
 *
 *  @return the current total SCIP time in seconds.
 */
EXTERN
SCIP_Real SCIPgetTotalTime(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the current solving time in seconds
 *
 *  @return the current solving time in seconds.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetSolvingTime(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the current reading time in seconds
 *
 *  @return the current reading time in seconds.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_PROBLEM
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetReadingTime(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the current presolving time in seconds
 *
 *  @return the current presolving time in seconds.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetPresolvingTime(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** gets the time need to solve the first LP in the root node
 *
 *  @return the solving time for the first LP in the root node in seconds.
 *
 *  @pre This method can be called if SCIP is in one of the following stages:
 *       - \ref SCIP_STAGE_TRANSFORMING
 *       - \ref SCIP_STAGE_TRANSFORMED
 *       - \ref SCIP_STAGE_INITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVING
 *       - \ref SCIP_STAGE_EXITPRESOLVE
 *       - \ref SCIP_STAGE_PRESOLVED
 *       - \ref SCIP_STAGE_INITSOLVE
 *       - \ref SCIP_STAGE_SOLVING
 *       - \ref SCIP_STAGE_SOLVED
 *
 *  See \ref SCIP_Stage "SCIP_STAGE" for a complete list of all possible solving stages.
 */
EXTERN
SCIP_Real SCIPgetFirstLPTime(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@} */




/*
 * numeric values and comparisons
 */

/**@name Numerical Methods */
/**@{ */


/** returns value treated as zero
 *
 *  @return value treated as zero
 */
EXTERN
SCIP_Real SCIPepsilon(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns value treated as zero for sums of floating point values
 *
 *  @return value treated as zero for sums of floating point values
 */
EXTERN
SCIP_Real SCIPsumepsilon(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns feasibility tolerance for constraints
 *
 *  @return feasibility tolerance for constraints
 */
EXTERN
SCIP_Real SCIPfeastol(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns primal feasibility tolerance of LP solver
 *
 *  @return primal feasibility tolerance of LP solver
 */
EXTERN
SCIP_Real SCIPlpfeastol(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns feasibility tolerance for reduced costs
 *
 *  @return feasibility tolerance for reduced costs
 */
EXTERN
SCIP_Real SCIPdualfeastol(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns convergence tolerance used in barrier algorithm
 *
 *  @return convergence tolerance used in barrier algorithm
 */
EXTERN
SCIP_Real SCIPbarrierconvtol(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** return the cutoff bound delta
 *
 *  @return cutoff bound data
 */
EXTERN
SCIP_Real SCIPcutoffbounddelta(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** sets the feasibility tolerance for constraints
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPchgFeastol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             feastol             /**< new feasibility tolerance for constraints */
   );

/** sets the primal feasibility tolerance of LP solver
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPchgLpfeastol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             lpfeastol,          /**< new primal feasibility tolerance of LP solver */
   SCIP_Bool             printnewvalue       /**< should "numerics/lpfeastol = ..." be printed? */
   );

/** sets the feasibility tolerance for reduced costs
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPchgDualfeastol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             dualfeastol         /**< new feasibility tolerance for reduced costs */
   );

/** sets the convergence tolerance used in barrier algorithm
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPchgBarrierconvtol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             barrierconvtol      /**< new convergence tolerance used in barrier algorithm */
   );

/** marks that some limit parameter was changed */
EXTERN
void SCIPmarkLimitChanged(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns value treated as infinity */
EXTERN
SCIP_Real SCIPinfinity(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** checks, if values are in range of epsilon */
EXTERN
SCIP_Bool SCIPisEQ(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if val1 is (more than epsilon) lower than val2 */
EXTERN
SCIP_Bool SCIPisLT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if val1 is not (more than epsilon) greater than val2 */
EXTERN
SCIP_Bool SCIPisLE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if val1 is (more than epsilon) greater than val2 */
EXTERN
SCIP_Bool SCIPisGT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if val1 is not (more than epsilon) lower than val2 */
EXTERN
SCIP_Bool SCIPisGE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if value is (positive) infinite */
EXTERN
SCIP_Bool SCIPisInfinity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to be compared against infinity */
   );

/** checks, if value is in range epsilon of 0.0 */
EXTERN
SCIP_Bool SCIPisZero(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks, if value is greater than epsilon */
EXTERN
SCIP_Bool SCIPisPositive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks, if value is lower than -epsilon */
EXTERN
SCIP_Bool SCIPisNegative(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks, if value is integral within epsilon */
EXTERN
SCIP_Bool SCIPisIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks whether the product val * scalar is integral in epsilon scaled by scalar */
EXTERN
SCIP_Bool SCIPisScalingIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val,                /**< unscaled value to check for scaled integrality */
   SCIP_Real             scalar              /**< value to scale val with for checking for integrality */
   );

/** checks, if given fractional part is smaller than epsilon */
EXTERN
SCIP_Bool SCIPisFracIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** rounds value + epsilon down to the next integer */
EXTERN
SCIP_Real SCIPfloor(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** rounds value - epsilon up to the next integer */
EXTERN
SCIP_Real SCIPceil(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** rounds value to the nearest integer with epsilon tolerance */
EXTERN
SCIP_Real SCIPround(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** returns fractional part of value, i.e. x - floor(x) in epsilon tolerance */
EXTERN
SCIP_Real SCIPfrac(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to return fractional part for */
   );

/** checks, if values are in range of sumepsilon */
EXTERN
SCIP_Bool SCIPisSumEQ(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if val1 is (more than sumepsilon) lower than val2 */
EXTERN
SCIP_Bool SCIPisSumLT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if val1 is not (more than sumepsilon) greater than val2 */
EXTERN
SCIP_Bool SCIPisSumLE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if val1 is (more than sumepsilon) greater than val2 */
EXTERN
SCIP_Bool SCIPisSumGT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if val1 is not (more than sumepsilon) lower than val2 */
EXTERN
SCIP_Bool SCIPisSumGE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if value is in range sumepsilon of 0.0 */
EXTERN
SCIP_Bool SCIPisSumZero(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks, if value is greater than sumepsilon */
EXTERN
SCIP_Bool SCIPisSumPositive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks, if value is lower than -sumepsilon */
EXTERN
SCIP_Bool SCIPisSumNegative(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks, if relative difference of values is in range of feasibility tolerance */
EXTERN
SCIP_Bool SCIPisFeasEQ(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference val1 and val2 is lower than feasibility tolerance */
EXTERN
SCIP_Bool SCIPisFeasLT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of val1 and val2 is not greater than feasibility tolerance */
EXTERN
SCIP_Bool SCIPisFeasLE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of val1 and val2 is greater than feastol */
EXTERN
SCIP_Bool SCIPisFeasGT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of val1 and val2 is not lower than -feastol */
EXTERN
SCIP_Bool SCIPisFeasGE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if value is in range feasibility tolerance of 0.0 */
EXTERN
SCIP_Bool SCIPisFeasZero(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks, if value is greater than feasibility tolerance */
EXTERN
SCIP_Bool SCIPisFeasPositive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks, if value is lower than -feasibility tolerance */
EXTERN
SCIP_Bool SCIPisFeasNegative(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks, if value is integral within the LP feasibility bounds */
EXTERN
SCIP_Bool SCIPisFeasIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks, if given fractional part is smaller than feastol */
EXTERN
SCIP_Bool SCIPisFeasFracIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** rounds value + feasibility tolerance down to the next integer */
EXTERN
SCIP_Real SCIPfeasFloor(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** rounds value - feasibility tolerance up to the next integer */
EXTERN
SCIP_Real SCIPfeasCeil(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** rounds value - feasibility tolerance up to the next integer in feasibility tolerance */
EXTERN
SCIP_Real SCIPfeasRound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** returns fractional part of value, i.e. x - floor(x) */
EXTERN
SCIP_Real SCIPfeasFrac(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   );

/** checks, if the given new lower bound is tighter (w.r.t. bound strengthening epsilon) than the old one */
EXTERN
SCIP_Bool SCIPisLbBetter(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             newlb,              /**< new lower bound */
   SCIP_Real             oldlb,              /**< old lower bound */
   SCIP_Real             oldub               /**< old upper bound */
   );

/** checks, if the given new upper bound is tighter (w.r.t. bound strengthening epsilon) than the old one */
SCIP_Bool SCIPisUbBetter(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             newub,              /**< new upper bound */
   SCIP_Real             oldlb,              /**< old lower bound */
   SCIP_Real             oldub               /**< old upper bound */
   );

/** checks, if relative difference of values is in range of epsilon */
EXTERN
SCIP_Bool SCIPisRelEQ(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of val1 and val2 is lower than epsilon */
EXTERN
SCIP_Bool SCIPisRelLT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of val1 and val2 is not greater than epsilon */
EXTERN
SCIP_Bool SCIPisRelLE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of val1 and val2 is greater than epsilon */
EXTERN
SCIP_Bool SCIPisRelGT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of val1 and val2 is not lower than -epsilon */
EXTERN
SCIP_Bool SCIPisRelGE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of values is in range of sumepsilon */
EXTERN
SCIP_Bool SCIPisSumRelEQ(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of val1 and val2 is lower than sumepsilon */
EXTERN
SCIP_Bool SCIPisSumRelLT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of val1 and val2 is not greater than sumepsilon */
EXTERN
SCIP_Bool SCIPisSumRelLE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of val1 and val2 is greater than sumepsilon */
EXTERN
SCIP_Bool SCIPisSumRelGT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** checks, if relative difference of val1 and val2 is not lower than -sumepsilon */
EXTERN
SCIP_Bool SCIPisSumRelGE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   );

/** Checks, if an iteratively updated value is reliable or should be recomputed from scratch.
 *  This is useful, if the value, e.g., the activity of a linear constraint or the pseudo objective value, gets a high
 *  absolute value during the optimization process which is later reduced significantly. In this case, the last digits
 *  were canceled out when increasing the value and are random after decreasing it.
 *  We do not consider the cancellations which can occur during increasing the absolute value because they just cannot
 *  be expressed using fixed precision floating point arithmetic, anymore.
 *  In order to get more reliable values, the idea is to always store the last reliable value, where increasing the
 *  absolute of the value is viewed as preserving reliability. Then, after each update, the new absolute value can be
 *  compared against the last reliable one with this method, checking whether it was decreased by a factor of at least
 *  "lp/recompfac" and should be recomputed.
 */
EXTERN
SCIP_Bool SCIPisUpdateUnreliable(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             newvalue,           /**< new value after update */
   SCIP_Real             oldvalue            /**< old value, i.e., last reliable value */
   );

/** checks, if value is huge and should be handled separately (e.g., in activity computation) */
EXTERN
SCIP_Bool SCIPisHugeValue(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to be checked whether it is huge */
   );

/** returns the minimum value that is regarded as huge and should be handled separately (e.g., in activity computation) */
EXTERN
SCIP_Real SCIPgetHugeValue(
   SCIP*                 scip                /**< SCIP data structure */
   );

#ifdef NDEBUG

/* In optimized mode, the function calls are overwritten by defines to reduce the number of function calls and
 * speed up the algorithms.
 */

#define SCIPinfinity(scip)                        SCIPsetInfinity((scip)->set)
#define SCIPisInfinity(scip, val)                 SCIPsetIsInfinity((scip)->set, val)        
#define SCIPisEQ(scip, val1, val2)                SCIPsetIsEQ((scip)->set, val1, val2)       
#define SCIPisLT(scip, val1, val2)                SCIPsetIsLT((scip)->set, val1, val2)       
#define SCIPisLE(scip, val1, val2)                SCIPsetIsLE((scip)->set, val1, val2)       
#define SCIPisGT(scip, val1, val2)                SCIPsetIsGT((scip)->set, val1, val2)       
#define SCIPisGE(scip, val1, val2)                SCIPsetIsGE((scip)->set, val1, val2)       
#define SCIPisZero(scip, val)                     SCIPsetIsZero((scip)->set, val)            
#define SCIPisPositive(scip, val)                 SCIPsetIsPositive((scip)->set, val)        
#define SCIPisNegative(scip, val)                 SCIPsetIsNegative((scip)->set, val)        
#define SCIPisIntegral(scip, val)                 SCIPsetIsIntegral((scip)->set, val)
#define SCIPisScalingIntegral(scip, val, scalar)  SCIPsetIsScalingIntegral((scip)->set, val, scalar)
#define SCIPisFracIntegral(scip, val)             SCIPsetIsFracIntegral((scip)->set, val)
#define SCIPfloor(scip, val)                      SCIPsetFloor((scip)->set, val)
#define SCIPceil(scip, val)                       SCIPsetCeil((scip)->set, val)
#define SCIPround(scip, val)                      SCIPsetRound((scip)->set, val)
#define SCIPfrac(scip, val)                       SCIPsetFrac((scip)->set, val)
                                                                                
#define SCIPisSumEQ(scip, val1, val2)             SCIPsetIsSumEQ((scip)->set, val1, val2)    
#define SCIPisSumLT(scip, val1, val2)             SCIPsetIsSumLT((scip)->set, val1, val2)    
#define SCIPisSumLE(scip, val1, val2)             SCIPsetIsSumLE((scip)->set, val1, val2)    
#define SCIPisSumGT(scip, val1, val2)             SCIPsetIsSumGT((scip)->set, val1, val2)    
#define SCIPisSumGE(scip, val1, val2)             SCIPsetIsSumGE((scip)->set, val1, val2)    
#define SCIPisSumZero(scip, val)                  SCIPsetIsSumZero((scip)->set, val)         
#define SCIPisSumPositive(scip, val)              SCIPsetIsSumPositive((scip)->set, val)     
#define SCIPisSumNegative(scip, val)              SCIPsetIsSumNegative((scip)->set, val)     
                                                                                
#define SCIPisFeasEQ(scip, val1, val2)            SCIPsetIsFeasEQ((scip)->set, val1, val2)   
#define SCIPisFeasLT(scip, val1, val2)            SCIPsetIsFeasLT((scip)->set, val1, val2)   
#define SCIPisFeasLE(scip, val1, val2)            SCIPsetIsFeasLE((scip)->set, val1, val2)   
#define SCIPisFeasGT(scip, val1, val2)            SCIPsetIsFeasGT((scip)->set, val1, val2)   
#define SCIPisFeasGE(scip, val1, val2)            SCIPsetIsFeasGE((scip)->set, val1, val2)   
#define SCIPisFeasZero(scip, val)                 SCIPsetIsFeasZero((scip)->set, val)        
#define SCIPisFeasPositive(scip, val)             SCIPsetIsFeasPositive((scip)->set, val)    
#define SCIPisFeasNegative(scip, val)             SCIPsetIsFeasNegative((scip)->set, val)    
#define SCIPisFeasIntegral(scip, val)             SCIPsetIsFeasIntegral((scip)->set, val)
#define SCIPisFeasFracIntegral(scip, val)         SCIPsetIsFeasFracIntegral((scip)->set, val)
#define SCIPfeasFloor(scip, val)                  SCIPsetFeasFloor((scip)->set, val)
#define SCIPfeasCeil(scip, val)                   SCIPsetFeasCeil((scip)->set, val)
#define SCIPfeasRound(scip, val)                  SCIPsetFeasRound((scip)->set, val)
#define SCIPfeasFrac(scip, val)                   SCIPsetFeasFrac((scip)->set, val)

                                                                                
#define SCIPisLbBetter(scip, newlb, oldlb, oldub) SCIPsetIsLbBetter(scip->set, newlb, oldlb, oldub)
#define SCIPisUbBetter(scip, newub, oldlb, oldub) SCIPsetIsUbBetter(scip->set, newub, oldlb, oldub)
                                                                                
#define SCIPisRelEQ(scip, val1, val2)             SCIPsetIsRelEQ((scip)->set, val1, val2)    
#define SCIPisRelLT(scip, val1, val2)             SCIPsetIsRelLT((scip)->set, val1, val2)    
#define SCIPisRelLE(scip, val1, val2)             SCIPsetIsRelLE((scip)->set, val1, val2)    
#define SCIPisRelGT(scip, val1, val2)             SCIPsetIsRelGT((scip)->set, val1, val2)    
#define SCIPisRelGE(scip, val1, val2)             SCIPsetIsRelGE((scip)->set, val1, val2)    
                                                                                
#define SCIPisSumRelEQ(scip, val1, val2)          SCIPsetIsSumRelEQ((scip)->set, val1, val2) 
#define SCIPisSumRelLT(scip, val1, val2)          SCIPsetIsSumRelLT((scip)->set, val1, val2) 
#define SCIPisSumRelLE(scip, val1, val2)          SCIPsetIsSumRelLE((scip)->set, val1, val2) 
#define SCIPisSumRelGT(scip, val1, val2)          SCIPsetIsSumRelGT((scip)->set, val1, val2) 
#define SCIPisSumRelGE(scip, val1, val2)          SCIPsetIsSumRelGE((scip)->set, val1, val2) 

#define SCIPisUpdateUnreliable(scip, newval, oldval) SCIPsetIsUpdateUnreliable((scip)->set, newval, oldval)
#define SCIPisHugeValue(scip, val) SCIPsetIsHugeValue((scip)->set, val)
#define SCIPgetHugeValue(scip) SCIPsetGetHugeValue((scip)->set)
#endif

/** outputs a real number, or "+infinity", or "-infinity" to a file */
EXTERN
void SCIPprintReal(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_Real             val,                /**< value to print */
   int                   width,              /**< width of the field */
   int                   precision           /**< number of significant digits printed */
   );

/**@} */




/*
 * memory management
 */

/**@name Memory Management */
/**@{ */

#define SCIPallocMemory(scip,ptr)               ( (BMSallocMemory((ptr)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPallocMemoryArray(scip,ptr,num)      ( (BMSallocMemoryArray((ptr), (num)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPallocClearMemoryArray(scip,ptr,num) ( (BMSallocClearMemoryArray((ptr), (num)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPallocMemorySize(scip,ptr,size)      ( (BMSallocMemorySize((ptr), (size)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPreallocMemoryArray(scip,ptr,newnum) ( (BMSreallocMemoryArray((ptr), (newnum)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPreallocMemorySize(scip,ptr,newsize) ( (BMSreallocMemorySize((ptr), (newsize)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPduplicateMemory(scip, ptr, source)  ( (BMSduplicateMemory((ptr), (source)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPduplicateMemoryArray(scip, ptr, source, num) \
                                                     ( (BMSduplicateMemoryArray((ptr), (source), (num)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPfreeMemory(scip,ptr)                BMSfreeMemory(ptr)
#define SCIPfreeMemoryNull(scip,ptr)            BMSfreeMemoryNull(ptr)
#define SCIPfreeMemoryArray(scip,ptr)           BMSfreeMemoryArray(ptr)
#define SCIPfreeMemoryArrayNull(scip,ptr)       BMSfreeMemoryArrayNull(ptr)
#define SCIPfreeMemorySize(scip,ptr)            BMSfreeMemorySize(ptr)
#define SCIPfreeMemorySizeNull(scip,ptr)        BMSfreeMemorySizeNull(ptr)

#define SCIPallocBlockMemory(scip,ptr)          ( (BMSallocBlockMemory(SCIPblkmem(scip), (ptr)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPallocBlockMemoryArray(scip,ptr,num) ( (BMSallocBlockMemoryArray(SCIPblkmem(scip), (ptr), (num)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPallocBlockMemorySize(scip,ptr,size) ( (BMSallocBlockMemorySize(SCIPblkmem(scip), (ptr), (size)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPreallocBlockMemoryArray(scip,ptr,oldnum,newnum) \
                                                     ( (BMSreallocBlockMemoryArray(SCIPblkmem(scip), (ptr), (oldnum), (newnum)) \
                                                       == NULL) ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPreallocBlockMemorySize(scip,ptr,oldsize,newsize) \
                                                     ( (BMSreallocBlockMemorySize(SCIPblkmem(scip), (ptr), (oldsize), (newsize)) \
                                                       == NULL) ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPduplicateBlockMemory(scip, ptr, source) \
                                                     ( (BMSduplicateBlockMemory(SCIPblkmem(scip), (ptr), (source)) == NULL) \
                                                       ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPduplicateBlockMemoryArray(scip, ptr, source, num) \
                                                     ( (BMSduplicateBlockMemoryArray(SCIPblkmem(scip), (ptr), (source), (num)) \
                                                       == NULL) ? SCIP_NOMEMORY : SCIP_OKAY )
#define SCIPensureBlockMemoryArray(scip,ptr,arraysizeptr,minsize) \
                                                     ( (SCIPensureBlockMemoryArray_call((scip), (void**)(ptr), sizeof(**(ptr)), \
                                                        (arraysizeptr), (minsize))) )
#define SCIPfreeBlockMemory(scip,ptr)           BMSfreeBlockMemory(SCIPblkmem(scip), (ptr))
#define SCIPfreeBlockMemoryNull(scip,ptr)       BMSfreeBlockMemoryNull(SCIPblkmem(scip), (ptr))
#define SCIPfreeBlockMemoryArray(scip,ptr,num)  BMSfreeBlockMemoryArray(SCIPblkmem(scip), (ptr), (num))
#define SCIPfreeBlockMemoryArrayNull(scip,ptr,num) \
                                                     BMSfreeBlockMemoryArrayNull(SCIPblkmem(scip), (ptr), (num))
#define SCIPfreeBlockMemorySize(scip,ptr,size)  BMSfreeBlockMemorySize(SCIPblkmem(scip), (ptr), (size))
#define SCIPfreeBlockMemorySizeNull(scip,ptr,size) \
                                                     BMSfreeBlockMemorySizeNull(SCIPblkmem(scip), (ptr), (size))

#define SCIPallocBuffer(scip,ptr)               SCIPallocBufferSize(scip, (void**)(ptr), (int)sizeof(**(ptr)))
#define SCIPallocBufferArray(scip,ptr,num)      SCIPallocBufferSize(scip, (void**)(ptr), (num)*(int)sizeof(**(ptr)))
#define SCIPreallocBufferArray(scip,ptr,num)    SCIPreallocBufferSize(scip, (void**)(ptr), (num)*(int)sizeof(**(ptr)))
#define SCIPduplicateBuffer(scip,ptr,source)    SCIPduplicateBufferSize(scip, (void**)(ptr), source, (int)sizeof(**(ptr)))
#define SCIPduplicateBufferArray(scip,ptr,source,num) \
                                                     SCIPduplicateBufferSize(scip, (void**)(ptr), source, (num)*(int)sizeof(**(ptr)))
#define SCIPfreeBuffer(scip,ptr)                SCIPfreeBufferSize(scip, (void**)(ptr), 0)
#define SCIPfreeBufferNull(scip,ptr)            { if( *(ptr) != NULL ) SCIPfreeBuffer(scip, ptr); }
#define SCIPfreeBufferArray(scip,ptr)           SCIPfreeBufferSize(scip, (void**)(ptr), 0)
#define SCIPfreeBufferArrayNull(scip,ptr)       { if( *(ptr) != NULL ) SCIPfreeBufferArray(scip, ptr); }


/** returns block memory to use at the current time
 *
 *  @return the block memory to use at the current time.
 */
EXTERN
BMS_BLKMEM* SCIPblkmem(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the total number of bytes used in block memory
 *
 *  @return the total number of bytes used in block memory.
 */
EXTERN
SCIP_Longint SCIPgetMemUsed(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the estimated number of bytes used by external software, e.g., the LP solver
 *
 *  @return the estimated number of bytes used by external software, e.g., the LP solver.
 */
EXTERN
SCIP_Longint SCIPgetMemExternEstim(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** calculate memory size for dynamically allocated arrays
 *
 *  @return the memory size for dynamically allocated arrays.
 */
EXTERN
int SCIPcalcMemGrowSize(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   num                 /**< minimum number of entries to store */
   );

/** extends a dynamically allocated block memory array to be able to store at least the given number of elements;
 *  use SCIPensureBlockMemoryArray() define to call this method!
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPensureBlockMemoryArray_call(
   SCIP*                 scip,               /**< SCIP data structure */
   void**                arrayptr,           /**< pointer to dynamically sized array */
   size_t                elemsize,           /**< size in bytes of each element in array */
   int*                  arraysize,          /**< pointer to current array size */
   int                   minsize             /**< required minimal array size */
   );

/** gets a memory buffer with at least the given size
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPallocBufferSize(
   SCIP*                 scip,               /**< SCIP data structure */
   void**                ptr,                /**< pointer to store the buffer */
   int                   size                /**< required size in bytes of buffer */
   );

/** allocates a memory buffer with at least the given size and copies the given memory into the buffer
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPduplicateBufferSize(
   SCIP*                 scip,               /**< SCIP data structure */
   void**                ptr,                /**< pointer to store the buffer */
   const void*           source,             /**< memory block to copy into the buffer */
   int                   size                /**< required size in bytes of buffer */
   );

/** reallocates a memory buffer to at least the given size
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPreallocBufferSize(
   SCIP*                 scip,               /**< SCIP data structure */
   void**                ptr,                /**< pointer to the buffer */
   int                   size                /**< required size in bytes of buffer */
   );

/** frees a memory buffer */
EXTERN
void SCIPfreeBufferSize(
   SCIP*                 scip,               /**< SCIP data structure */
   void**                ptr,                /**< pointer to the buffer */
   int                   dummysize           /**< used to get a safer define for SCIPfreeBuffer() and SCIPfreeBufferArray() */
   );

/** prints output about used memory */
EXTERN
void SCIPprintMemoryDiagnostic(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@} */




/*
 * dynamic arrays
 */

/**@name Dynamic Arrays */
/**@{ */

/** creates a dynamic array of real values
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPcreateRealarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY**      realarray           /**< pointer to store the real array */
   );

/** frees a dynamic array of real values
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPfreeRealarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY**      realarray           /**< pointer to the real array */
   );

/** extends dynamic array to be able to store indices from minidx to maxidx
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPextendRealarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray,          /**< dynamic real array */
   int                   minidx,             /**< smallest index to allocate storage for */
   int                   maxidx              /**< largest index to allocate storage for */
   );

/** clears a dynamic real array
 *
 *  @return clears a dynamic real array
 */
EXTERN
SCIP_RETCODE SCIPclearRealarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray           /**< dynamic real array */
   );

/** gets value of entry in dynamic array */
EXTERN
SCIP_Real SCIPgetRealarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray,          /**< dynamic real array */
   int                   idx                 /**< array index to get value for */
   );

/** sets value of entry in dynamic array
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetRealarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray,          /**< dynamic real array */
   int                   idx,                /**< array index to set value for */
   SCIP_Real             val                 /**< value to set array index to */
   );

/** increases value of entry in dynamic array
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPincRealarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray,          /**< dynamic real array */
   int                   idx,                /**< array index to increase value for */
   SCIP_Real             incval              /**< value to increase array index */
   );

/** returns the minimal index of all stored non-zero elements
 *
 *  @return the minimal index of all stored non-zero elements
 */
EXTERN
int SCIPgetRealarrayMinIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray           /**< dynamic real array */
   );

/** returns the maximal index of all stored non-zero elements
 *
 *  @return the maximal index of all stored non-zero elements
 */
EXTERN
int SCIPgetRealarrayMaxIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray           /**< dynamic real array */
   );

/** creates a dynamic array of int values
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPcreateIntarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY**       intarray            /**< pointer to store the int array */
   );

/** frees a dynamic array of int values
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPfreeIntarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY**       intarray            /**< pointer to the int array */
   );

/** extends dynamic array to be able to store indices from minidx to maxidx
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPextendIntarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray,           /**< dynamic int array */
   int                   minidx,             /**< smallest index to allocate storage for */
   int                   maxidx              /**< largest index to allocate storage for */
   );

/** clears a dynamic int array
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPclearIntarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray            /**< dynamic int array */
   );

/** gets value of entry in dynamic array
 *
 *  @return value of entry in dynamic array
 */
EXTERN
int SCIPgetIntarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray,           /**< dynamic int array */
   int                   idx                 /**< array index to get value for */
   );

/** sets value of entry in dynamic array
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetIntarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray,           /**< dynamic int array */
   int                   idx,                /**< array index to set value for */
   int                   val                 /**< value to set array index to */
   );

/** increases value of entry in dynamic array
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPincIntarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray,           /**< dynamic int array */
   int                   idx,                /**< array index to increase value for */
   int                   incval              /**< value to increase array index */
   );

/** returns the minimal index of all stored non-zero elements
 *
 *  @return the minimal index of all stored non-zero elements
 */
EXTERN
int SCIPgetIntarrayMinIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray            /**< dynamic int array */
   );

/** returns the maximal index of all stored non-zero elements
 *
 *  @return the maximal index of all stored non-zero elements
 */
EXTERN
int SCIPgetIntarrayMaxIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray            /**< dynamic int array */
   );

/** creates a dynamic array of bool values
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPcreateBoolarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY**      boolarray           /**< pointer to store the bool array */
   );

/** frees a dynamic array of bool values
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPfreeBoolarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY**      boolarray           /**< pointer to the bool array */
   );

/** extends dynamic array to be able to store indices from minidx to maxidx
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPextendBoolarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray,          /**< dynamic bool array */
   int                   minidx,             /**< smallest index to allocate storage for */
   int                   maxidx              /**< largest index to allocate storage for */
   );

/** clears a dynamic bool array
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPclearBoolarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray           /**< dynamic bool array */
   );

/** gets value of entry in dynamic array
 *
 *  @return value of entry in dynamic array at position idx
 */
EXTERN
SCIP_Bool SCIPgetBoolarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray,          /**< dynamic bool array */
   int                   idx                 /**< array index to get value for */
   );

/** sets value of entry in dynamic array
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetBoolarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray,          /**< dynamic bool array */
   int                   idx,                /**< array index to set value for */
   SCIP_Bool             val                 /**< value to set array index to */
   );

/** returns the minimal index of all stored non-zero elements
 *
 *  @return the minimal index of all stored non-zero elements
 */
EXTERN
int SCIPgetBoolarrayMinIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray           /**< dynamic bool array */
   );

/** returns the maximal index of all stored non-zero elements
 *
 *  @return the maximal index of all stored non-zero elements
 */
EXTERN
int SCIPgetBoolarrayMaxIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray           /**< dynamic bool array */
   );

/** creates a dynamic array of pointers
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPcreatePtrarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY**       ptrarray            /**< pointer to store the int array */
   );

/** frees a dynamic array of pointers
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPfreePtrarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY**       ptrarray            /**< pointer to the int array */
   );

/** extends dynamic array to be able to store indices from minidx to maxidx
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPextendPtrarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray,           /**< dynamic int array */
   int                   minidx,             /**< smallest index to allocate storage for */
   int                   maxidx              /**< largest index to allocate storage for */
   );

/** clears a dynamic pointer array
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPclearPtrarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray            /**< dynamic int array */
   );

/** gets value of entry in dynamic array */
EXTERN
void* SCIPgetPtrarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray,           /**< dynamic int array */
   int                   idx                 /**< array index to get value for */
   );

/** sets value of entry in dynamic array
 *
 *  @return \ref SCIP_OKAY is returned if everything worked. Otherwise a suitable error code is passed. See \ref
 *          SCIP_Retcode "SCIP_RETCODE" for a complete list of error codes.
 */
EXTERN
SCIP_RETCODE SCIPsetPtrarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray,           /**< dynamic int array */
   int                   idx,                /**< array index to set value for */
   void*                 val                 /**< value to set array index to */
   );

/** returns the minimal index of all stored non-zero elements
 *
 *  @return the minimal index of all stored non-zero elements
 */
EXTERN
int SCIPgetPtrarrayMinIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray            /**< dynamic ptr array */
   );

/** returns the maximal index of all stored non-zero elements
 *
 *  @return the maximal index of all stored non-zero elements
 */
EXTERN
int SCIPgetPtrarrayMaxIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray            /**< dynamic ptr array */
   );

#ifdef NDEBUG

/* In optimized mode, the function calls are overwritten by defines to reduce the number of function calls and
 * speed up the algorithms.
 */

#define SCIPcreateRealarray(scip, realarray)                    SCIPrealarrayCreate(realarray, SCIPblkmem(scip))
#define SCIPfreeRealarray(scip, realarray)                      SCIPrealarrayFree(realarray)
#define SCIPextendRealarray(scip, realarray, minidx, maxidx)    SCIPrealarrayExtend(realarray, (scip)->set, minidx, maxidx)
#define SCIPclearRealarray(scip, realarray)                     SCIPrealarrayClear(realarray)
#define SCIPgetRealarrayVal(scip, realarray, idx)               SCIPrealarrayGetVal(realarray, idx)
#define SCIPsetRealarrayVal(scip, realarray, idx, val)          SCIPrealarraySetVal(realarray, (scip)->set, idx, val)
#define SCIPincRealarrayVal(scip, realarray, idx, incval)       SCIPrealarrayIncVal(realarray, scip->set, idx, incval)
#define SCIPgetRealarrayMinIdx(scip, realarray)                 SCIPrealarrayGetMinIdx(realarray)
#define SCIPgetRealarrayMaxIdx(scip, realarray)                 SCIPrealarrayGetMaxIdx(realarray)

#define SCIPcreateIntarray(scip, intarray)                      SCIPintarrayCreate(intarray, SCIPblkmem(scip))
#define SCIPfreeIntarray(scip, intarray)                        SCIPintarrayFree(intarray)
#define SCIPextendIntarray(scip, intarray, minidx, maxidx)      SCIPintarrayExtend(intarray, (scip)->set, minidx, maxidx)
#define SCIPclearIntarray(scip, intarray)                       SCIPintarrayClear(intarray)
#define SCIPgetIntarrayVal(scip, intarray, idx)                 SCIPintarrayGetVal(intarray, idx)
#define SCIPsetIntarrayVal(scip, intarray, idx, val)            SCIPintarraySetVal(intarray, (scip)->set, idx, val)
#define SCIPincIntarrayVal(scip, intarray, idx, incval)         SCIPintarrayIncVal(intarray, scip->set, idx, incval)
#define SCIPgetIntarrayMinIdx(scip, intarray)                   SCIPintarrayGetMinIdx(intarray)
#define SCIPgetIntarrayMaxIdx(scip, intarray)                   SCIPintarrayGetMaxIdx(intarray)

#define SCIPcreateBoolarray(scip, boolarray)                    SCIPboolarrayCreate(boolarray, SCIPblkmem(scip))
#define SCIPfreeBoolarray(scip, boolarray)                      SCIPboolarrayFree(boolarray)
#define SCIPextendBoolarray(scip, boolarray, minidx, maxidx)    SCIPboolarrayExtend(boolarray, (scip)->set, minidx, maxidx)
#define SCIPclearBoolarray(scip, boolarray)                     SCIPboolarrayClear(boolarray)
#define SCIPgetBoolarrayVal(scip, boolarray, idx)               SCIPboolarrayGetVal(boolarray, idx)
#define SCIPsetBoolarrayVal(scip, boolarray, idx, val)          SCIPboolarraySetVal(boolarray, (scip)->set, idx, val)
#define SCIPgetBoolarrayMinIdx(scip, boolarray)                 SCIPboolarrayGetMinIdx(boolarray)
#define SCIPgetBoolarrayMaxIdx(scip, boolarray)                 SCIPboolarrayGetMaxIdx(boolarray)

#define SCIPcreatePtrarray(scip, ptrarray)                      SCIPptrarrayCreate(ptrarray, SCIPblkmem(scip))
#define SCIPfreePtrarray(scip, ptrarray)                        SCIPptrarrayFree(ptrarray)
#define SCIPextendPtrarray(scip, ptrarray, minidx, maxidx)      SCIPptrarrayExtend(ptrarray, (scip)->set, minidx, maxidx)
#define SCIPclearPtrarray(scip, ptrarray)                       SCIPptrarrayClear(ptrarray)
#define SCIPgetPtrarrayVal(scip, ptrarray, idx)                 SCIPptrarrayGetVal(ptrarray, idx)
#define SCIPsetPtrarrayVal(scip, ptrarray, idx, val)            SCIPptrarraySetVal(ptrarray, (scip)->set, idx, val)
#define SCIPgetPtrarrayMinIdx(scip, ptrarray)                   SCIPptrarrayGetMinIdx(ptrarray)
#define SCIPgetPtrarrayMaxIdx(scip, ptrarray)                   SCIPptrarrayGetMaxIdx(ptrarray)

#endif

/**@} */
#ifdef __cplusplus
}
#endif

#endif

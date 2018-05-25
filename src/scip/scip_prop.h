/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2018 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   scip_prop.h
 * @ingroup PUBLICCOREAPI
 * @brief  public methods for propagator plugins
 * @author Tobias Achterberg
 * @author Timo Berthold
 * @author Thorsten Koch
 * @author Alexander Martin
 * @author Marc Pfetsch
 * @author Kati Wolter
 * @author Gregor Hendel
 * @author Robert Lion Gottwald
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_SCIP_PROP_H__
#define __SCIP_SCIP_PROP_H__


#include "scip/def.h"
#include "scip/type_lp.h"
#include "scip/type_prop.h"
#include "scip/type_result.h"
#include "scip/type_retcode.h"
#include "scip/type_scip.h"
#include "scip/type_timing.h"
#include "scip/type_var.h"

/* In debug mode, we include the SCIP's structure in scip.c, such that no one can access
 * this structure except the interface methods in scip.c.
 * In optimized mode, the structure is included in scip.h, because some of the methods
 * are implemented as defines for performance reasons (e.g. the numerical comparisons).
 * Additionally, the internal "set.h" is included, such that the defines in set.h are
 * available in optimized mode.
 */
#ifdef NDEBUG
#include "scip/struct_scip.h"
#include "scip/struct_stat.h"
#include "scip/set.h"
#include "scip/tree.h"
#include "scip/misc.h"
#include "scip/var.h"
#include "scip/cons.h"
#include "scip/solve.h"
#include "scip/debug.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**@addtogroup PublicPropagatorMethods
 *
 * @{
 */

/** creates a propagator and includes it in SCIP.
 *

 *  @note method has all propagator callbacks as arguments and is thus changed every time a new
 *        callback is added in future releases; consider using SCIPincludePropBasic() and setter functions
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
   SCIP_PROPTIMING       timingmask,         /**< positions in the node solving loop where propagator should be executed */
   int                   presolpriority,     /**< presolving priority of the propagator (>= 0: before, < 0: after constraint handlers) */
   int                   presolmaxrounds,    /**< maximal number of presolving rounds the propagator participates in (-1: no limit) */
   SCIP_PRESOLTIMING     presoltiming,       /**< timing mask of the propagator's presolving method */
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
   SCIP_PRESOLTIMING     presoltiming        /**< timing mask of the propagator's presolving method */
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

/* @} */

#ifdef __cplusplus
}
#endif

#endif

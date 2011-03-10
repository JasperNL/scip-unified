/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2011 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   prop_xyz.c
 * @ingroup PROPAGATORS
 * @brief  xyz propagator
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "scip/prop_xyz.h"


#define PROP_NAME              "xyz"
#define PROP_DESC              "propagator template"
#define PROP_PRIORITY                 0
#define PROP_FREQ                    10
#define PROP_DELAY                FALSE /**< should propagation method be delayed, if other propagators found reductions? */




/*
 * Data structures
 */

/* TODO: fill in the necessary propagator data */

/** propagator data */
struct SCIP_PropData
{
};




/*
 * Local methods
 */

/* put your local methods here, and declare them static */




/*
 * Callback methods of propagator
 */

/* TODO: Implement all necessary propagator methods. The methods with an #if 0 ... #else #define ... are optional */


/** copy method for propagator plugins (called when SCIP copies plugins) */
#if 0
static
SCIP_DECL_PROPCOPY(propCopyXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz propagator not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/
 
   return SCIP_OKAY;
}
#else
#define propCopyXyz NULL
#endif

/** destructor of propagator to free user data (called when SCIP is exiting) */
#if 0
static
SCIP_DECL_PROPFREE(propFreeXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz propagator not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define propFreeXyz NULL
#endif


/** initialization method of propagator (called after problem was transformed) */
#if 0
static
SCIP_DECL_PROPINIT(propInitXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz propagator not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define propInitXyz NULL
#endif


/** deinitialization method of propagator (called before transformed problem is freed) */
#if 0
static
SCIP_DECL_PROPEXIT(propExitXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz propagator not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define propExitXyz NULL
#endif


/** solving process initialization method of propagator (called when branch and bound process is about to begin) */
#if 0
static
SCIP_DECL_PROPINITSOL(propInitsolXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz propagator not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define propInitsolXyz NULL
#endif


/** solving process deinitialization method of propagator (called before branch and bound process data is freed) */
#if 0
static
SCIP_DECL_PROPEXITSOL(propExitsolXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz propagator not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define propExitsolXyz NULL
#endif


/** execution method of propagator */
static
SCIP_DECL_PROPEXEC(propExecXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz propagator not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}


/** propagation conflict resolving method of propagator */
static
SCIP_DECL_PROPRESPROP(propRespropXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz propagator not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}




/*
 * propagator specific interface methods
 */

/** creates the xyz propagator and includes it in SCIP */
SCIP_RETCODE SCIPincludePropXyz(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_PROPDATA* propdata;

   /* create xyz propagator data */
   propdata = NULL;
   /* TODO: (optional) create propagator specific data here */

   /* include propagator */
   SCIP_CALL( SCIPincludeProp(scip, PROP_NAME, PROP_DESC, PROP_PRIORITY, PROP_FREQ, PROP_DELAY,
         propCopyXyz,
         propFreeXyz, propInitXyz, propExitXyz, 
         propInitsolXyz, propExitsolXyz, propExecXyz, propRespropXyz,
         propdata) );

   /* add xyz propagator parameters */
   /* TODO: (optional) add propagator specific parameters with SCIPaddTypeParam() here */

   return SCIP_OKAY;
}

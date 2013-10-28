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

/**@file   unittest-relax.c
 * @brief  unit test for checking setters on scip.c
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <scip/scip.h>
#include "scip/scipdefplugins.h"
#include "relax_unittest.h"
#include <string.h>

/** macro to check the return of tests
 *
 *  @note assumes the existence of SCIP_RETCODE retcode
 */
#define CHECK_TEST(x)                            \
   do                                            \
   {                                             \
      retcode = (x);                             \
      if( retcode != SCIP_OKAY )                 \
      {                                          \
         printf("Unit test " #x " failed\n");    \
         SCIPprintError(retcode);                \
         return -1;                              \
      }                                          \
   }                                             \
   while( FALSE )

/** macro to check the value of a 'getter' and 'value'
  *
  */
#define CHECK_GET(getter, value)   \
   do                              \
   {                               \
      if( getter != value )        \
         return SCIP_ERROR;        \
   }                               \
   while(FALSE)



/* help methods */
static
SCIP_RETCODE initProb(
   SCIP*                 scip                /**< SCIP instance */
   )
{
   SCIP_CONS* cons;
   SCIP_VAR* xvar;
   SCIP_VAR* yvar;
   SCIP_VAR* vars[2];
   SCIP_Real vals[2];

   /* create variables */
   SCIP_CALL( SCIPcreateVarBasic(scip, &xvar, "x", -SCIPinfinity(scip), SCIPinfinity(scip), 1.0, SCIP_VARTYPE_INTEGER) );
   SCIP_CALL( SCIPcreateVarBasic(scip, &yvar, "y", -SCIPinfinity(scip), SCIPinfinity(scip), -1.0, SCIP_VARTYPE_INTEGER) );

   SCIP_CALL( SCIPaddVar(scip, xvar) );
   SCIP_CALL( SCIPaddVar(scip, yvar) );

   /* create inequalities */
   vars[0] = xvar;
   vars[1] = yvar;

   vals[0] = 1.0;
   vals[1] = -1.0;

   SCIP_CALL( SCIPcreateConsBasicLinear(scip, &cons, "lower", 2, vars, vals, 0.25, 0.75) );
   SCIP_CALL( SCIPaddCons(scip, cons) );

   SCIP_CALL( SCIPreleaseCons(scip, &cons) );
   SCIP_CALL( SCIPreleaseVar(scip, &xvar) );
   SCIP_CALL( SCIPreleaseVar(scip, &yvar) );

   return SCIP_OKAY;
}


/* Check methods */
static
SCIP_RETCODE relaxCheckName(SCIP_RELAX* relax)
{
   const char* name = "relax-unittest";
   CHECK_GET( strcmp(SCIPrelaxGetName(relax),name) , 0 );

   return SCIP_OKAY;
}

static
SCIP_RETCODE relaxCheckDesc(SCIP_RELAX* relax)
{
   const char* name = "relaxator template";
   CHECK_GET( strcmp(SCIPrelaxGetDesc(relax),name) , 0 );

   return SCIP_OKAY;
}

static
SCIP_RETCODE relaxCheckPriority(SCIP_RELAX* relax)
{
   int priority;
   priority = 101;
   CHECK_GET( SCIPrelaxGetPriority(relax), priority );

   return SCIP_OKAY;
}

static
SCIP_RETCODE relaxCheckFreq(SCIP_RELAX* relax)
{
   int freq;
   freq = 2;
   CHECK_GET( SCIPrelaxGetFreq(relax), freq );

   return SCIP_OKAY;
}

/*@todo how to check this? */
static
SCIP_RETCODE relaxCheckSetupTime(SCIP_RELAX* relax)
{
   if( SCIPrelaxGetSetupTime(relax) < 0 )
      return SCIP_ERROR;
   return SCIP_OKAY;
}

/*@todo how to check this? */
static
SCIP_RETCODE relaxCheckTime(SCIP_RELAX* relax)
{
   if( SCIPrelaxGetTime(relax) < 0 )
      return SCIP_ERROR;
   return SCIP_OKAY;
}

/*@todo how to check this? */
static
SCIP_RETCODE relaxCheckNCalls(SCIP_RELAX* relax)
{
   return SCIP_OKAY;
}


/** main function */
int
main(
   int                        argc,
   char**                     argv
   )
{
   SCIP_RETCODE retcode;
   SCIP* scip;
   SCIP_RELAX* relax;

   /*********
    * Setup *
    *********/
   scip = NULL;

   /* initialize SCIP */
   SCIP_CALL( SCIPcreate(&scip) );

   /* include default SCIP plugins */
   SCIP_CALL( SCIPincludeDefaultPlugins(scip) );

   /* include binpacking reader */
   SCIP_CALL( SCIPincludeRelaxUnittest(scip) );

   /* store the relaxertor */
   relax = SCIPgetRelaxs(scip)[0];

   /* create a problem and disable the presolver */
   SCIP_CALL( SCIPcreateProbBasic(scip, "problem") );


   /*********
    * Tests *
    *********/
   CHECK_TEST( relaxCheckName(relax) );
   CHECK_TEST( relaxCheckDesc(relax) );
   CHECK_TEST( relaxCheckPriority(relax) );
   CHECK_TEST( relaxCheckFreq(relax) );
   CHECK_TEST( relaxCheckSetupTime(relax) );
   CHECK_TEST( relaxCheckTime(relax) );
   CHECK_TEST( relaxCheckNCalls(relax) );
   /********************
    * Deinitialization *
    ********************/

   SCIP_CALL( SCIPfree(&scip) );

   BMScheckEmptyMemory();

   printf("All tests passed\n");
   return 0;
}

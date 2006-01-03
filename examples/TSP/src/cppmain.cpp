/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2006 Tobias Achterberg                              */
/*                                                                           */
/*                  2002-2006 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License.             */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: cppmain.cpp,v 1.7 2006/01/03 12:22:40 bzfpfend Exp $"

/**@file   cppmain.cpp
 * @brief  main file for C++ TSP example using SCIP as a callable library
 * @author Tobias Achterberg
 * @author Timo Berthold
 */

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <iostream>

/* include SCIP components */
#include "objscip/objscip.h"


extern "C"
{
#include "scip/scipdefplugins.h"
}

/* include TSP specific components */
#include "ReaderTSP.h"
#include "ConshdlrSubtour.h"
#include "HeurFarthestInsert.h"
#include "Heur2opt.h"
#include "EventhdlrNewSol.h"

using namespace scip;
using namespace tsp;
using namespace std;

static
SCIP_RETCODE readParams(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename            /**< parameter file name, or NULL */
   )
{
   if( filename != NULL )
   {
      if( SCIPfileExists(filename) )
      {
         std::cout << "reading parameter file <" << filename << ">" << std::endl;
         SCIP_CALL( SCIPreadParams(scip, filename) );
      }
      else
         std::cout << "parameter file <" << filename << "> not found - using default parameters" << std::endl;
   }
   else if( SCIPfileExists("sciptsp.set") )
   {
      std::cout << "reading parameter file <sciptsp.set>" << std::endl;
      SCIP_CALL( SCIPreadParams(scip, "sciptsp.set") );
   }

   return SCIP_OKAY;
}

static
SCIP_RETCODE fromCommandLine(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename            /**< input file name */
   )
{
   /********************
    * Problem Creation *
    ********************/

   std::cout << std::endl << "read problem <" << filename << ">" << std::endl;
   std::cout << "============" << std::endl << std::endl;
   SCIP_CALL( SCIPreadProb(scip, filename) );


   /*******************
    * Problem Solving *
    *******************/

   /* solve problem */
   std::cout << "solve problem" << std::endl;
   std::cout << "=============" << std::endl;
   SCIP_CALL( SCIPsolve(scip) );

   std::cout << std::endl << "primal solution:" << std::endl;
   std::cout << "================" << std::endl << std::endl;
   SCIP_CALL( SCIPprintBestSol(scip, NULL, FALSE) );


   /**************
    * Statistics *
    **************/

   std::cout << std::endl << "Statistics" << std::endl;
   std::cout << "==========" << std::endl << std::endl;

   SCIP_CALL( SCIPprintStatistics(scip, NULL) );

   return SCIP_OKAY;
}

static
SCIP_RETCODE interactive(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   /* start user interactive mode */
   SCIP_CALL( SCIPstartInteraction(scip) );

   return SCIP_OKAY;
}

static
SCIP_RETCODE runSCIP(
   int                   argc,
   char**                argv
   )
{
   SCIP* scip = NULL;


   /***********************
    * Version information *
    ***********************/

   SCIPprintVersion(NULL);
   std::cout << std::endl;


   /*********
    * Setup *
    *********/

   /* initialize SCIP */
   SCIP_CALL( SCIPcreate(&scip) );

   /* include TSP specific plugins */
   SCIP_CALL( SCIPincludeObjReader(scip, new ReaderTSP(scip), TRUE) );
   SCIP_CALL( SCIPincludeObjConshdlr(scip, new ConshdlrSubtour(), TRUE) ); 
   SCIP_CALL( SCIPincludeObjEventhdlr(scip, new EventhdlrNewSol(), TRUE) );
   SCIP_CALL( SCIPincludeObjHeur(scip, new HeurFarthestInsert(), TRUE) );
   SCIP_CALL( SCIPincludeObjHeur(scip, new Heur2opt(), TRUE) );
   
   /* include default SCIP plugins */
   SCIP_CALL( SCIPincludeDefaultPlugins(scip) );


   /**************
    * Parameters *
    **************/

   if( argc >= 3 )
   {
      SCIP_CALL( readParams(scip, argv[2]) );
   }
   else
   {
      SCIP_CALL( readParams(scip, NULL) );
   }
   /*CHECK_OKAY( SCIPwriteParams(scip, "sciptsp.set", TRUE) );*/


   /**************
    * Start SCIP *
    **************/

   if( argc >= 2 )
   {
      SCIP_CALL( fromCommandLine(scip, argv[1]) );
   }
   else
   {
      printf("\n");

      SCIP_CALL( interactive(scip) );
   }

   
   /********************
    * Deinitialization *
    ********************/

   SCIP_CALL( SCIPfree(&scip) );

   BMScheckEmptyMemory();

   return SCIP_OKAY;
}

int
main(
   int                   argc,
   char**                argv
   )
{
   SCIP_RETCODE retcode;

   retcode = runSCIP(argc, argv);
   if( retcode != SCIP_OKAY )
   {
      SCIPprintError(retcode, stderr);
      return -1;
   }

   return 0;
}

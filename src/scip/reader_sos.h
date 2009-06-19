/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2009 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: reader_sos.h,v 1.1.2.1 2009/06/19 07:53:49 bzfwolte Exp $"

/**@file   reader_sos.h
 * @brief  specially ordered set (SOS) file reader
 * @author Marc Pfetsch
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_READER_SOS_H__
#define __SCIP_READER_SOS_H__


#include "scip/scip.h"


/** includes the SOS file reader into SCIP */
extern
SCIP_RETCODE SCIPincludeReaderSOS(
   SCIP*                 scip                /**< SCIP data structure */
   );

#endif

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2017 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   reader_cflp.h
 * @brief  Cflp problem reader file reader
 * @author Stephen J. Maher
 *
 * This file implements the reader/parser used to read the cflp input data. For more details see \ref CFLP_READER.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_READER_CFLP_H__
#define __SCIP_READER_CFLP_H__


#include "scip/scip.h"


/** includes the cflp file reader into SCIP */
extern
SCIP_RETCODE SCIPincludeReaderCflp(
   SCIP*                 scip                /**< SCIP data structure */
   );

#endif

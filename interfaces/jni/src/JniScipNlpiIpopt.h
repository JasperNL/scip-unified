/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class de_zib_jscip_nativ_jni_JniScipNlpiIpopt */

#ifndef _Included_de_zib_jscip_nativ_jni_JniScipNlpiIpopt
#define _Included_de_zib_jscip_nativ_jni_JniScipNlpiIpopt
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     de_zib_jscip_nativ_jni_JniScipNlpiIpopt
 * Method:    createNlpSolverIpopt
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_de_zib_jscip_nativ_jni_JniScipNlpiIpopt_createNlpSolverIpopt
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipNlpiIpopt
 * Method:    getSolverNameIpopt
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_de_zib_jscip_nativ_jni_JniScipNlpiIpopt_getSolverNameIpopt
  (JNIEnv *, jobject);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipNlpiIpopt
 * Method:    getSolverDescIpopt
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_de_zib_jscip_nativ_jni_JniScipNlpiIpopt_getSolverDescIpopt
  (JNIEnv *, jobject);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipNlpiIpopt
 * Method:    isIpoptAvailableIpopt
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_de_zib_jscip_nativ_jni_JniScipNlpiIpopt_isIpoptAvailableIpopt
  (JNIEnv *, jobject);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipNlpiIpopt
 * Method:    getIpoptApplicationPointerIpopt
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_de_zib_jscip_nativ_jni_JniScipNlpiIpopt_getIpoptApplicationPointerIpopt
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipNlpiIpopt
 * Method:    getNlpiOracleIpopt
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_de_zib_jscip_nativ_jni_JniScipNlpiIpopt_getNlpiOracleIpopt
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipNlpiIpopt
 * Method:    setModifiedDefaultSettingsIpopt
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_de_zib_jscip_nativ_jni_JniScipNlpiIpopt_setModifiedDefaultSettingsIpopt
  (JNIEnv *, jobject, jlong, jstring);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipNlpiIpopt
 * Method:    LapackDsyev
 * Signature: (ZI[D[D)V
 */
JNIEXPORT void JNICALL Java_de_zib_jscip_nativ_jni_JniScipNlpiIpopt_LapackDsyev
  (JNIEnv *, jobject, jboolean, jint, jdoubleArray, jdoubleArray);

#ifdef __cplusplus
}
#endif
#endif

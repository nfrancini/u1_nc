AC_INIT([u1_nc], [0.1], [nicco.francini@libero.it])


AM_INIT_AUTOMAKE(subdir-objects)


AC_CONFIG_HEADERS([config.h])


AC_PROG_CC


AC_CHECK_HEADERS([math.h], [], [AC_MSG_ERROR([Required header math.h not found.])])

AC_CHECK_HEADERS([stdio.h], [], [AC_MSG_ERROR([Required header stdio.h not found.])])

AC_CHECK_HEADERS([stdlib.h], [], [AC_MSG_ERROR([Required header stdlib.h not found.])])

AC_CHECK_HEADERS([time.h], [], [AC_MSG_ERROR([Required header time.h not found.])])

AC_CHECK_HEADERS([stddef.h], [], [AC_MSG_ERROR([Required header stddef.h not found.])])

AC_CHECK_HEADERS([complex.h], [], [AC_MSG_ERROR([Required header complex.h not found.])])

AC_CHECK_HEADERS([errno.h], [], [AC_MSG_ERROR([Required header errno.h not found.])])

AC_CHECK_HEADERS([fenv.h], [], [AC_MSG_ERROR([Required header fenv.h not found.])])

AC_CHECK_HEADERS([sys/types.h], [], [AC_MSG_ERROR([Required header sys/types.h not found.])])

AC_CHECK_HEADERS([sys/stat.h], [], [AC_MSG_ERROR([Required header sys/stat.h not found.])])


AC_SEARCH_LIBS([sqrt], [m], [], [AC_MSG_ERROR([Required c mathematical library not found.])])


AC_ARG_VAR(N, the number of flavours (default 2))

AC_ARG_VAR(D, spacetime dimensionality (default 3))


if test x${N} = x; then
  AC_MSG_WARN([
   *****
   No chosen value for N: using the default N=4
   *****
   ])
  AC_DEFINE_UNQUOTED(N, 4, [Number of flavours])
else
  AC_DEFINE_UNQUOTED(N, ${N}, [Number of flavours])
fi

if test x${D} = x; then
  AC_MSG_WARN([
   *****
   No chosen value for D: using the default D=3
   *****
   ])
  AC_DEFINE_UNQUOTED(D, 3, [Spacetime dimensionality])
else
  AC_DEFINE_UNQUOTED(D, ${D}, [Spacetime dimensionality])
fi


AC_CONFIG_FILES([Makefile])

AC_OUTPUT

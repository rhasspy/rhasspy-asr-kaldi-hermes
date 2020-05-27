AC_INIT([rhasspy-asr-kaldi-hermes], [0.1.1], [mike@rhasspy.org])
AC_CONFIG_MACRO_DIR([m4])
PC_INIT([3.7.0])

dnl ---------------------------------------------------------------------------

AC_PREFIX_DEFAULT([$PWD/.venv])

dnl Template files to write
AC_CONFIG_FILES([Makefile rhasspy-asr-kaldi-hermes])

AC_CANONICAL_HOST
AC_PROG_INSTALL
AC_PROG_MKDIR_P
AC_PROG_CC
AC_PROG_CXX

dnl ---------------------------------------------------------------------------

is_virtualenv_enabled=yes
is_dependency_check_enabled=yes

dnl An in-place install does not include Rhasspy Python modules, since they will
dnl be used directly from source.
is_in_place=no

dnl --------------
dnl Speech to Text
dnl --------------

is_kaldi_enabled=UNKNOWN
kaldi_from=prebuilt

dnl --------------
dnl Training Tools
dnl --------------

is_training_enabled=yes

is_opengrm_enabled=UNKNOWN
opengrm_from=prebuilt

is_phonetisaurus_enabled=UNKNOWN
phonetisaurus_from=prebuilt

dnl Set architectures based on host CPU
AS_CASE([$host_cpu],
    [armv6l],[
        dnl ARM 32-bit v6 (Pi 1/0)
        docker_arch=armv6

        is_kaldi_enabled=no
        kaldi_from=NONE

        is_opengrm_enabled=yes

        is_phonetisaurus_enabled=yes
    ],
    [armv7l],[
        dnl ARM 32-bit v7 (Pi 2/3/4)
        docker_arch=armv7

        is_kaldi_enabled=yes

        is_opengrm_enabled=yes

        is_phonetisaurus_enabled=yes
    ],
    [aarch64],[
        dnl ARM 64-bit (Pi 3/4)
        docker_arch=arm64

        is_kaldi_enabled=yes

        is_opengrm_enabled=yes

        is_phonetisaurus_enabled=yes
    ],
    [x86_64],[
        dnl x86_64 compatible
        docker_arch=amd64

        is_kaldi_enabled=yes

        is_opengrm_enabled=yes

        is_phonetisaurus_enabled=yes
    ])

dnl ---------------------------------------------------------------------------

AC_ARG_ENABLE([in-place],
    AS_HELP_STRING([--enable-in-place],
        [do not install Rhasspy Python modules in virtual environment (will be used from source)]))

precompiled_binaries_enabled=yes

AC_ARG_ENABLE([precompiled-binaries],
    AS_HELP_STRING([--disable-precompiled-binaries],
        [build dependencies from source instead of downloading prebuilt binaries]))

dnl ---------------------------------------------------------------------------
dnl Python virtual environment
dnl ---------------------------------------------------------------------------

AC_ARG_ENABLE([virtualenv],
    [AS_HELP_STRING([--disable-virtualenv],
        [don't create a Python virtual environment at prefix])])

AC_ARG_VAR([VIRTUALENV_FLAGS], [flags to pass to when creating virtual environment])

dnl ---------------------------------------------------------------------------
dnl Training Tools
dnl ---------------------------------------------------------------------------

dnl Opengrm
AC_ARG_ENABLE([opengrm],
    AS_HELP_STRING([--disable-opengrm],
        [disable installation of opengrm language modeling toolkit]))

dnl Phonetisaurus
AC_ARG_ENABLE([phonetisaurus],
    AS_HELP_STRING([--disable-phonetisaurus],
        [disable installation of phonetisaurus grapheme to phoneme tool]))

dnl Training meta
AC_ARG_ENABLE([training],
    AS_HELP_STRING([--disable-training],
        [disable installation of training tools]))

dnl ---------------------------------------------------------------------------
dnl Speech to Text Systems
dnl ---------------------------------------------------------------------------

dnl Kaldi
AC_ARG_ENABLE([kaldi],
    AS_HELP_STRING([--disable-kaldi],
        [disable installation of Kaldi speech to text system]))

dnl ---------------------------------------------------------------------------
dnl Resolve Settings
dnl ---------------------------------------------------------------------------

AS_CASE([$enable_training],
        [no],[
            is_opengrm_enabled=no
            is_phonetisaurus_enabled=no
            is_kenlm_enabled=no
        ])

dnl Override with enable/disable options (non-meta)
AS_CASE([$enable_kaldi],
        [yes],[is_kaldi_enabled=yes],
        [no],[is_kaldi_enabled=no])

AS_CASE([$enable_opengrm],
    [yes],[is_opengrm_enabled=yes],
    [no],[is_opengrm_enabled=no])

AS_CASE([$enable_phonetisaurus],
    [yes],[is_phonetisaurus_enabled=yes],
    [no],[is_phonetisaurus_enabled=no])

dnl Determine prebuilt/source
AS_CASE([$enable_precompiled_binaries],
        [no],[
            kaldi_from=source
            opengrm_from=source
            phonetisaurus_from=source
        ])

dnl In-place install
AS_CASE([$enable_in_place],
    [yes],[is_in_place=yes],
    [no],[is_in_place=no])

dnl ---------------------------------------------------------------------------
dnl Extra Dependency Checks
dnl ---------------------------------------------------------------------------

AC_ARG_ENABLE([dependency-check],
    AS_HELP_STRING([--disable-dependency-check],
        [disable checking for external dependencies]),
    [is_dependency_check_enabled=$enableval])

AS_IF([test "x$is_dependency_check_enabled" = xyes], [

dnl kaldi needs ATLAS
AS_IF([test "x$is_kaldi_enabled" = xyes],
    [
        AC_CHECK_LIB(atlas, ATL_xerbla, [], [
            AC_MSG_ERROR([ATLAS is required for Kaldi (libatlas-base-dev package)])
        ])
    ])

dnl kaldi needs a C++ compiler (and a ton of other things)
AS_IF([test "x${is_kaldi_enabled}x${kaldi_from}" = xyesxsource],
    [
        AS_IF([test "x$CXX" = x], [
            AC_MSG_ERROR([C++ compiler is required for Kaldi (build-essential package)])
        ])
    ])

])  dnl is_dependency_check_enabled

dnl ---------------------------------------------------------------------------
dnl Summary
dnl ---------------------------------------------------------------------------

dnl Prefix is NONE for some reason instead of default value
summary_prefix=$prefix
AS_IF([test "x$summary_prefix" = xNONE], [
    summary_prefix=$PWD/.venv
])

AS_ECHO(["
configuration summary:

architecture: ${host_cpu}/${docker_arch}
prefix: ${summary_prefix}
virtualenv: ${is_virtualenv_enabled}
in-place: ${is_in_place}"])

AS_ECHO(["
speech to text:"])

AS_IF([test "x$is_kaldi_enabled" = xyes],
AS_ECHO(["  kaldi: ${is_kaldi_enabled} (${docker_arch}, ${kaldi_from})"]),
AS_ECHO(["  kaldi: ${is_kaldi_enabled}"]))

AS_ECHO(["
training:"])

AS_IF([test "x$is_opengrm_enabled" = xyes],
AS_ECHO(["  opengrm: ${is_opengrm_enabled} (${docker_arch}, ${opengrm_from})"]),
AS_ECHO(["  opengrm: ${is_opengrm_enabled}"]))

AS_IF([test "x$is_phonetisaurus_enabled" = xyes],
AS_ECHO(["  phonetisaurus: ${is_phonetisaurus_enabled} (${docker_arch}, ${phonetisaurus_from})"]),
AS_ECHO(["  phonetisaurus: ${is_phonetisaurus_enabled}"]))

AS_ECHO([""])

dnl ---------------------------------------------------------------------------
dnl Output
dnl ---------------------------------------------------------------------------

AC_SUBST([VIRTUALENV], [$is_virtualenv_enabled])

AC_SUBST([DOCKER_ARCH], [$docker_arch])

AC_SUBST([ENABLE_KALDI], [$is_kaldi_enabled])
AC_SUBST([KALDI_FROM], [$kaldi_from])

AC_SUBST([ENABLE_OPENGRM], [$is_opengrm_enabled])
AC_SUBST([OPENGRM_FROM], [$opengrm_from])

AC_SUBST([ENABLE_PHONETISAURUS], [$is_phonetisaurus_enabled])
AC_SUBST([PHONETISAURUS_FROM], [$phonetisaurus_from])

AC_SUBST([IN_PLACE], [$is_in_place])

AC_OUTPUT

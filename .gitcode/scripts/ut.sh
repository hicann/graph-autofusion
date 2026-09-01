#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set +e
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
sudo update-alternatives --set gcc /usr/bin/gcc-14
gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
pip3 install -r super_kernel/requirements-dev.txt

touch test1.log
coverage_save="true"
if [ "${target_branch}" == "master" ] || [ "${target_branch}" == "develop" ]; then
	case "${ut_type}" in
		UT_Test_Python_superkernel)
			bash build.sh -u -c --impl=py --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource" -j20 -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		UT_Test_superkernel)
			bash build.sh -u -c --impl=cpp --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		UT_Test_autofuse_framework)
			bash build.sh -u -c --module=autofuse_framework --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		UT_Test_autofuse_ascendc_api)
			bash build.sh -u -c --module=autofuse_ascendc_api --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		ST_Test_Python_superkernel)
			bash build.sh -u -c --impl=py --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		ST_Test_autofuse_framework)
			bash build.sh -s -c --module=autofuse_framework --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			coverage_save="false"
			;;
		ST_Test_autofuse_ascendc_api)
			bash build.sh -s -c --module=autofuse_ascendc_api --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt -j20 | tee test_ascendc_api.log
			ret=$?
			coverage_save="false"
			;;
		ST_Test_autofuse_e2e)
			bash build.sh -s -c  --module=autofuse_e2e --cann_3rd_lib_path="/home/jenkins/opensource" -j20 -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			coverage_save="false"
			;;
        *)
            echo "Skip UT test execution for ${ut_type} on non-master branch"
            exit 0
            ;;
	esac
else
	case "${ut_type}" in
		UT_Test_Python_superkernel)
			bash build.sh -u -c --impl=py --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			;;
		UT_Test_superkernel)
			bash build.sh -u -c --impl=cpp --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			;;
		UT_Test_autofuse_framework)
			bash build.sh -u -c --module=autofuse_framework --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			;;
		UT_Test_autofuse_ascendc_api)
			bash build.sh -u -c --module=autofuse_ascendc_api --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			;;
		ST_Test_Python_superkernel)
			bash build.sh -u -c --impl=py --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			;;
		ST_Test_autofuse_framework)
			bash build.sh -s -c --module=autofuse_framework --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			coverage_save="false"
			;;
		ST_Test_autofuse_ascendc_api)
			bash build.sh -s -c --module=autofuse_ascendc_api --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			coverage_save="false"
			;;
		ST_Test_autofuse_e2e)
			bash build.sh -s -c  --module=autofuse_e2e --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			coverage_save="false"
			;;
        *)
            echo "Skip UT test execution for ${ut_type} on non-master branch"
            exit 0
            ;;
	esac
fi


if [ $ret -ne 200 ] && [ $ret -ne 0 ]; then
    echo "run ut fail"
    exit 1
fi
if [ $ret -eq 0 ]; then
    if [ "$coverage_save" = "true" ];then
    echo "ut_process=coverage" >> $ATOMGIT_OUTPUT
    else
    echo "ut_process=ut_cov" >> $ATOMGIT_OUTPUT
    fi
fi
exit 0

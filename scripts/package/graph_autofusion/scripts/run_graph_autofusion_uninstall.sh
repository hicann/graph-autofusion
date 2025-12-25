#!/bin/bash
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

username="$(id -un)"
usergroup="$(id -gn)"
is_quiet=n
curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"
pkg_version_path="${curpath}/../version.info"

. "${common_func_path}"

if [ "$1" ]; then
    input_install_dir="${2}"
    common_parse_type="${3}"
    is_quiet="${4}"
    is_docker_install="${5}"  # 兼容跨版本调用保留参数
    docker_root="${6}"
    is_recreate_softlink="${7}"
    pkg_version_dir="${8}"
fi

if [ "${is_recreate_softlink}" = "y" ]; then
    recreate_softlink_option="--recreate-softlink"
else
    recreate_softlink_option=""
fi

if [ "x${docker_root}" != "x" ]; then
    common_parse_dir="${docker_root}${input_install_dir}"
else
    common_parse_dir="${input_install_dir}"
fi

get_version "pkg_version" "$pkg_version_path"
is_multi_version_pkg "pkg_is_multi_version" "$pkg_version_path"
if [ "$pkg_is_multi_version" = "true" ]; then
    common_parse_dir="$common_parse_dir/$pkg_version_dir"
fi

if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi
logfile="${log_dir}/ascend_install.log"

SOURCE_INSTALL_COMMON_PARSER_FILE="${common_parse_dir}/share/info/graph_autofusion/script/install_common_parser.sh"
SOURCE_FILELIST_FILE="${common_parse_dir}/share/info/graph_autofusion/script/filelist.csv"

install_info="${common_parse_dir}/share/info/graph_autofusion/ascend_install.info"

# 写日志
log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local log_type="$1"
    local log_msg="$2"
    local log_format="[graph_autofusion] [$cur_date] [$log_type]: $log_msg"
    if [ "$log_type" = "INFO" ]; then
        echo "$log_format"
    elif [ "$log_type" = "WARNING" ]; then
        echo "$log_format"
    elif [ "$log_type" = "ERROR" ]; then
        echo "$log_format"
    elif [ "$log_type" = "DEBUG" ]; then
        echo "$log_format" 1> /dev/null
    fi
    echo "$log_format" >> "$logfile"
}

##########################################################################
log "INFO" "step into run_graph_autofusion_uninstall.sh ......"
log "INFO" "uninstall target dir $common_parse_dir, type $common_parse_type."

if [ ! -d "$common_parse_dir/share/info/graph_autofusion" ]; then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir/share/info/graph_autofusion is not exist."
    exit 1
fi

new_uninstall() {
    if [ -f "${common_parse_dir}/share/info/graph_autofusion/data/version.info" ]; then
        log "INFO" "need to uninstall costmodel files."
        bash "${common_parse_dir}/share/info/graph_autofusion/data/script/install.sh" -- -- --uninstall --install-path="${common_parse_dir}"
    fi

    if [ ! -d "${common_parse_dir}/share/info/graph_autofusion" ]; then
        log "INFO" "no need to uninstall graph_autofusion files."
        return 0
    fi

    # 赋可写权限
    chmod +w -R "${SOURCE_INSTALL_COMMON_PARSER_FILE}"

    if [ "$pkg_is_multi_version" = "true" ]; then
        local package_db_info="$common_parse_dir/var/ascend_package_db.info"
        if [ -e "$package_db_info" ]; then
            local linux_path="$(realpath $common_parse_dir/..)"
            local arch_path="$(basename $linux_path)"
            local latest_path="$(realpath $linux_path/../..)/cann"
            local pkgs="$(cut -d'|' -f2 $package_db_info | sort -u)"
            if [ "$pkgs" = "graph_autofusion" ]; then
                if [ -L "$latest_path/$arch_path" ] && [ "$(realpath $linux_path)" = "$(realpath $latest_path/$arch_path)" ]; then
                    rm -f "$latest_path/$arch_path"
                fi
            elif [ -n "$pkgs" ] && [ -d "$latest_path" ]; then
                if [ ! -e "$latest_path/$arch_path" ] || [ -L "$latest_path/$arch_path" ]; then
                    ln -srfn "$linux_path" "$latest_path"
                fi
            fi
        fi
    fi

    # 执行卸载
    custom_options="--custom-options=--common-parse-dir=$common_parse_dir,--logfile=$logfile,--stage=uninstall,--quiet=$is_quiet"
    sh "$SOURCE_INSTALL_COMMON_PARSER_FILE" --package="graph_autofusion" --uninstall --username="$username" --usergroup="$usergroup" ${recreate_softlink_option} \
        --version=$pkg_version --version-dir=$pkg_version_dir --use-share-info \
        --docker-root="$docker_root" $custom_options "$common_parse_type" "$input_install_dir" "$SOURCE_FILELIST_FILE"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0090;ERR_DES:failed to uninstall package."
        return 1
    fi

    if [ -n "$latest_path" ] && [ -d "$latest_path" ] && [ "x$(ls -A $latest_path 2>&1)" = "x" ]; then
        rm -rf "$latest_path"
    fi

    return 0
}

new_uninstall
if [ $? -ne 0 ]; then
    exit 1
fi

exit 0

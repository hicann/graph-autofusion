set(ascendc_api_regbase_extend_src
        concat.h
        datacopy.h
        broadcast.h
        broadcast_extend_impl.h
        datacopy_nddma.h
        gather.h
        clipbyvalue.h
        utils.h
        logical_not.h
        logical_xor.h
        cast.h
        logical.h
        compare.h
        floor_div.h
        pow.h
        polygamma.h
        split.h
        where.h
        erf.h
        erfinv.h
        exp2.h
        log1p.h
        tanh.h
        sub.h
        div.h
        sign.h
        signbit.h
        reduce_init.h
        atan2.h
        airy_ai.h
        copy_sign.h
        erfcx.h
        expm.h
        fmod.h
        neg.h
        square.h
        softmax_af.h
        trunc_div.h
        remainder.h
        bessel_j_utils.h
        bessel_j0.h
        bessel_j1.h
        spherical_bessel_j0.h
        bessel_y0.h
        bessel_y1.h
        bucketize.h
        modified_bessel_i0.h
        i0e.h
        modified_bessel_i1.h
        i1e.h
        modified_bessel_k0.h
        modified_bessel_k1.h
        modified_bessel_utils.h
        scaled_modified_bessel_k0.h
        scaled_modified_bessel_k1.h
        laguerre_polynomial_l.h
        legendre_polynomial_p.h
        igamma.h
        igammac.h
        trigonometric_function_utils.h
        transpose.h
        frexp.h
        zeta.h
        ndtri.h
        shifted_chebyshev_polynomial_utils.h
        shifted_chebyshev_polynomial_t.h
        shifted_chebyshev_polynomial_u.h
        shifted_chebyshev_polynomial_v.h
        shifted_chebyshev_polynomial_w.h
        ndtr.h
        log_ndtr.h
        next_after.h
        i0.h
        hermite_polynomial_h.h
        hermite_polynomial_he.h
        igammac_helper/series.h
        igammac_helper/continued_fraction.h
        igammac_helper/asymptotic_series.h
        igammac_helper/series_complement.h
        hypot.h
        indirect_load_simd.h
        indirect_load_sk.h
        indirect_load_simt.h
        chebyshev_polynomial_utils.h
        chebyshev_polynomial_t.h
        chebyshev_polynomial_u.h
        chebyshev_polynomial_v.h
        chebyshev_polynomial_w.h
)

set(generated_regbase_files)

foreach(header ${ascendc_api_regbase_extend_src})
    string(REGEX REPLACE "\\.h$" "_reg_base.h" str_header ${header})
    get_filename_component(output_subdir ${str_header} DIRECTORY)
    if(output_subdir)
        set(make_dir_cmd COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${output_subdir})
    else()
        set(make_dir_cmd)
    endif()
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            ${make_dir_cmd}
            COMMAND cat ${header}
            | sed "'1i\\R\"===('"
            | sed "'$$a\\)===\"'"
            > ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
            DEPENDS ${header}
    )
    list(APPEND generated_regbase_files ${CMAKE_CURRENT_BINARY_DIR}/${str_header})
endforeach()

add_custom_target(generated_regbase_files ALL DEPENDS ${generated_regbase_files})

add_library(ascendc_api_regbase_extend INTERFACE)
target_include_directories(ascendc_api_regbase_extend INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(ascendc_api_regbase_extend generated_regbase_files)

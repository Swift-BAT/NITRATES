import numpy as np

from ..response.Materials import (
    CarbonFibre,
    SiC,
    AL,
    AG,
    ElecMix,
    ElecMixDense,
    AlHoney1,
    AlHoney2,
    AlHoney3,
    PB,
    TA,
    SN,
    CU,
)
from ..response.StructClasses import (
    Swift_Structure,
    Swift_Structure_Compound,
    Swift_Structure_wEmbededPolys,
    Swift_Structure_Shield,
    Swift_Structure_Sun_Shield,
    Swift_Structure_Manager,
)
from ..response.Polygons import Box_Polygon, Cylinder_Polygon, Cylinder_wHoles_Polygon


def get_full_struct_manager(Es=None, structs2ignore=[]):
    # UVOT stuff
    uvot_cent = (-28.95, 83.95, 47.392)

    uvot_baffe_cyl = Cylinder_Polygon(
        uvot_cent[0], uvot_cent[1], uvot_cent[2] + 6.194, 18.45, 18.95, 77.4
    )
    UVOT_baffe = Swift_Structure(uvot_baffe_cyl, CarbonFibre, Name="UVOTbaffe")

    uvot_elec_cyl = Cylinder_Polygon(
        uvot_cent[0], uvot_cent[1], uvot_cent[2] - 77.4, 0.0, 18.95, 6.194
    )
    UVOT_elec = Swift_Structure(uvot_elec_cyl, ElecMix, Name="UVOTelec")

    uvot_fast_cyl = Cylinder_Polygon(
        uvot_cent[0], uvot_cent[1], -32.201, 18.95, 20.05, 2.0
    )
    UVOT_fast = Swift_Structure(uvot_fast_cyl, AL, Name="UVOTfast")

    # XRT stuff
    xrt_cent = (25.4, 89.45, 47.392)
    xrt_r0 = 24.756
    xrt_r1 = 25.4
    xrt_half_length = 83.594

    XRT_baffe_cyl = Cylinder_Polygon(
        xrt_cent[0], xrt_cent[1], xrt_cent[2], xrt_r0, xrt_r1, xrt_half_length
    )
    XRT_baffe = Swift_Structure(XRT_baffe_cyl, CarbonFibre, Name="XRTbaffe")

    XRT_optics_cyl = Cylinder_Polygon(
        xrt_cent[0], xrt_cent[1], xrt_cent[2] - 17.907, 10.345, 15.0, 50.0
    )
    XRT_optics = Swift_Structure(XRT_optics_cyl, SiC, Name="XRToptics")

    XRT_fast_cyl = Cylinder_Polygon(
        xrt_cent[0], xrt_cent[1], -31.201, 25.4, 26.177, 5.0
    )
    XRT_fast = Swift_Structure(XRT_fast_cyl, AL, Name="XRTfast")


    # Star tracker

    star_track_cent = (25.4+31.855+2., 88.45-11.594+6.0, 113.486)

    star_track_elec_box0_cent = (star_track_cent[0] + 0, star_track_cent[1] + 9, star_track_cent[2] - 9)
    star_track_elec_box1_cent = (star_track_cent[0] + 0, star_track_cent[1] - 9, star_track_cent[2] - 9)


    star_track_elecs_half_dims = [8.5, 8.5, 8.5]

    star_track_elecs_box0 = Box_Polygon(
        star_track_elecs_half_dims[0],
        star_track_elecs_half_dims[1],
        star_track_elecs_half_dims[2],
        np.array(star_track_elec_box0_cent),
    )

    star_track_elecs_box1 = Box_Polygon(
        star_track_elecs_half_dims[0],
        star_track_elecs_half_dims[1],
        star_track_elecs_half_dims[2],
        np.array(star_track_elec_box1_cent),
    )

    Star_Track_Elecs_Box0 = Swift_Structure(star_track_elecs_box0, ElecMixDense, Name='StarTrackElecBox0')
    Star_Track_Elecs_Box1 = Swift_Structure(star_track_elecs_box1, ElecMixDense, Name='StarTrackElecBox1')

    star_track_col_r0 = 5.0
    star_track_col_r1 = 5.7
    star_track_col_half_length = 9.0

    star_track_cyl0_cent = (star_track_cent[0] + 0, star_track_cent[1] + 9, star_track_cent[2] - 8.5)
    star_track_cyl1_cent = (star_track_cent[0] + 0, star_track_cent[1] - 9, star_track_cent[2] - 8.5)

    star_track_cyl0 = Cylinder_Polygon(star_track_cyl0_cent[0], star_track_cyl0_cent[1], star_track_cyl0_cent[2],\
                                    star_track_col_r0, star_track_col_r1, star_track_col_half_length)

    star_track_cyl1 = Cylinder_Polygon(star_track_cyl1_cent[0], star_track_cyl1_cent[1], star_track_cyl1_cent[2],\
                                    star_track_col_r0, star_track_col_r1, star_track_col_half_length)

    star_track_col0 = Swift_Structure(star_track_cyl0, CarbonFibre, Name='StarTrackColl0')
    star_track_col1 = Swift_Structure(star_track_cyl1, CarbonFibre, Name='StarTrackColl1')


    # Spacecraft Body
    body_pos = (0.0, 43.0, -198.701)
    body_r0 = 79.0
    body_r1 = 82.0
    body_half_height = 150.0

    body_cyl = Cylinder_Polygon(
        body_pos[0], body_pos[1], body_pos[2], body_r0, body_r1, body_half_height
    )
    Body = Swift_Structure(body_cyl, AL, Name="Body")

    # Obs Bench
    obs_bench_r0 = 0.0
    obs_bench_r1 = 124.5
    obs_bench_half_height = 5.25
    obs_bench_pos = (0.0, -20.0 + 63.0, -41.452)

    obs_bench = Cylinder_Polygon(
        obs_bench_pos[0],
        obs_bench_pos[1],
        obs_bench_pos[2],
        obs_bench_r0,
        obs_bench_r1,
        obs_bench_half_height,
    )

    obs_bench_cent_r0 = 0.0
    obs_bench_cent_r1 = 78.5
    obs_bench_cent_half_height = 5.05

    hole_cents = [
        [25.4, 20.0 + 25.45 + obs_bench_pos[1], obs_bench_pos[2]],
        [-28.95, 20.0 + 20.95 + obs_bench_pos[1], obs_bench_pos[2]],
    ]
    hole_rs = [25.4, 18.95]

    obs_bench_cent = Cylinder_wHoles_Polygon(
        obs_bench_pos[0],
        obs_bench_pos[1],
        obs_bench_pos[2],
        obs_bench_cent_r1,
        hole_cents,
        hole_rs,
        obs_bench_cent_half_height,
    )

    face_half_height = 0.1

    hole_cents[0][2] = obs_bench_pos[2] - 5.15
    hole_cents[1][2] = obs_bench_pos[2] - 5.15
    obs_bench_face0 = Cylinder_wHoles_Polygon(
        obs_bench_pos[0],
        obs_bench_pos[1],
        obs_bench_pos[2] - 5.15,
        obs_bench_r1,
        hole_cents,
        hole_rs,
        face_half_height,
    )

    hole_cents[0][2] = obs_bench_pos[2] + 5.15
    hole_cents[1][2] = obs_bench_pos[2] + 5.15
    obs_bench_face1 = Cylinder_wHoles_Polygon(
        obs_bench_pos[0],
        obs_bench_pos[1],
        obs_bench_pos[2] + 5.15,
        obs_bench_r1,
        hole_cents,
        hole_rs,
        face_half_height,
    )

    child_polys = [obs_bench_cent, obs_bench_face0, obs_bench_face1]
    material_list = [AlHoney3, AlHoney2, AL, AL]

    Obs_Bench = Swift_Structure_wEmbededPolys(
        obs_bench, child_polys, material_list, Name="ObsBench"
    )

    # Electronic Boxes
    ElecBox_half_dims = [15.0, 15.0, 20.0]
    Elecs_half_dims = [14.0, 19.0, 19.0]
    ElecBox_pos0 = (70.8, 83.05, -16.201)
    ElecBox_pos1 = (-70.8, 83.05, -16.201)

    Elec_box0 = Box_Polygon(
        ElecBox_half_dims[0],
        ElecBox_half_dims[1],
        ElecBox_half_dims[2],
        np.array(ElecBox_pos0),
    )
    Elec_box1 = Box_Polygon(
        ElecBox_half_dims[0],
        ElecBox_half_dims[1],
        ElecBox_half_dims[2],
        np.array(ElecBox_pos1),
    )

    Elecs_box0 = Box_Polygon(
        Elecs_half_dims[0],
        Elecs_half_dims[1],
        Elecs_half_dims[2],
        np.array(ElecBox_pos0),
    )
    Elecs_box1 = Box_Polygon(
        Elecs_half_dims[0],
        Elecs_half_dims[1],
        Elecs_half_dims[2],
        np.array(ElecBox_pos1),
    )

    ElecBox0 = Swift_Structure_wEmbededPolys(
        Elec_box0, [Elecs_box0], [AL, ElecMixDense], Name="ElecBox0"
    )
    ElecBox1 = Swift_Structure_wEmbededPolys(
        Elec_box1, [Elecs_box1], [AL, ElecMixDense], Name="ElecBox1"
    )

    # Stuff on BAT
    BATZ_offset = 35.799

    # PCB
    pcb_half_dims = [59.842, 16.510, 3.094]
    pcb_elec_half_dims = [59.5, 16.2, 2.7]
    pcb_pos0 = (0.0, 30.48, BATZ_offset - 17.304 - 32.612)
    pcb_pos1 = (0.0, -30.48, BATZ_offset - 17.304 - 32.612)
    print(pcb_pos0)

    PCB_box0 = Box_Polygon(
        pcb_half_dims[0], pcb_half_dims[1], pcb_half_dims[2], np.array(pcb_pos0)
    )
    PCBelec_box0 = Box_Polygon(
        pcb_elec_half_dims[0],
        pcb_elec_half_dims[1],
        pcb_elec_half_dims[2],
        np.array(pcb_pos0),
    )
    PCB0 = Swift_Structure_wEmbededPolys(
        PCB_box0, [PCBelec_box0], [AL, ElecMix], Name="PCB0"
    )

    PCB_box1 = Box_Polygon(
        pcb_half_dims[0], pcb_half_dims[1], pcb_half_dims[2], np.array(pcb_pos1)
    )
    PCBelec_box1 = Box_Polygon(
        pcb_elec_half_dims[0],
        pcb_elec_half_dims[1],
        pcb_elec_half_dims[2],
        np.array(pcb_pos1),
    )
    PCB1 = Swift_Structure_wEmbededPolys(
        PCB_box1, [PCBelec_box1], [AL, ElecMix], Name="PCB1"
    )

    # Detector Array Panel (DAP)
    DAP_half_dims = [81.915, 46.990, 3.175]
    dap_pos = (0.0, 0.0, (BATZ_offset - 11.035 - 32.612))

    DAP_box = Box_Polygon(
        DAP_half_dims[0], DAP_half_dims[1], DAP_half_dims[2], np.array(dap_pos)
    )

    DAPcore_half_dims = [81.915, 46.990, 3.073]
    DAPcore_box = Box_Polygon(
        DAPcore_half_dims[0],
        DAPcore_half_dims[1],
        DAPcore_half_dims[2],
        np.array(dap_pos),
    )

    DAP = Swift_Structure_wEmbededPolys(
        DAP_box, [DAPcore_box], [AL, AlHoney1], Name="DAP"
    )

    struct_list = [
        UVOT_baffe,
        UVOT_elec,
        UVOT_fast,
        XRT_baffe,
        XRT_optics,
        XRT_fast,
        # PCB0,
        # PCB1,
        Obs_Bench,
        DAP,
        # ElecBox0,
        # ElecBox1,
        Body,
        Star_Track_Elecs_Box0,
        Star_Track_Elecs_Box1,
        star_track_col0,
        star_track_col1
    ]

    if not "Shield" in structs2ignore:
        Shield = Swift_Structure_Shield()
        struct_list.append(Shield)
    if not "SunShield" in structs2ignore:
        SunShield = Swift_Structure_Sun_Shield()
        struct_list.append(SunShield)

    struct_manager = Swift_Structure_Manager()
    if Es is not None:
        struct_manager.set_energy_arr(Es)

    for struct in struct_list:
        struct_manager.add_struct(struct)

    return struct_manager

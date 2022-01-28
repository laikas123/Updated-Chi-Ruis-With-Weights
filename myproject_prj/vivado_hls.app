<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="myproject_prj" top="myproject">
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
    <files>
        <file name="../../tb_data" sc="0" tb="1" cflags="  -Wno-unknown-pragmas"/>
        <file name="../../firmware/weights" sc="0" tb="1" cflags="  -Wno-unknown-pragmas"/>
        <file name="../../myproject_test.cpp" sc="0" tb="1" cflags=" -std=c++0x -Wno-unknown-pragmas"/>
        <file name="firmware/myproject.cpp" sc="0" tb="false" cflags="-std=c++0x"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
</AutoPilot:project>


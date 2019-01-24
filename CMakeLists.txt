cmake_minimum_required(VERSION 3.12)
project(PAIGE)

set(CMAKE_CXX_STANDARD 14)

find_package(Shark REQUIRED)
include(${SHARK_USE_FILE})

add_executable(TestModel TestModel.cpp PAIGE_Loss.cpp PAIGE_Loss.h)


target_link_libraries( TestModel
        PRIVATE
        ${SHARK_LIBRARIES}
        )

add_executable(TrainModel TrainModel.cpp PAIGE_Loss.cpp PAIGE_Loss.h )

target_link_libraries(TrainModel
        PRIVATE
        ${SHARK_LIBRARIES}
        )


add_executable(AppendTrainModel AppendTrainModel.cpp PAIGE_Loss.cpp PAIGE_Loss.h)

target_link_libraries(AppendTrainModel
        PRIVATE
        ${SHARK_LIBRARIES}
        )

add_executable(TrainModel_with_ManyCSV TrainModel_with_ManyCSV.cpp PAIGE_Loss.cpp PAIGE_Loss.h)

target_link_libraries(TrainModel_with_ManyCSV
        PRIVATE
        ${SHARK_LIBRARIES}
        )


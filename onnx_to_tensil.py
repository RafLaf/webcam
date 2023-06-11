"""
Compile all models using tensil package, user_specified architecture and onnx
Save :
    - logs of the compilation in a csv file
    - tensil model files (.tmodel, .tarch, .tdata)
"""

import docker
import os
import csv
import argparse

# file paths

ONNX_PATH = "onnx/"
PATH_OUTPUT="tensil/"

# global variables
architecture = ""

parameters_list = {
    "data_type": "FP16BP8",
    "array_size": 8,
    "dram0_depth": 1048576,
    "dram1_depth": 1048576,
    "local_depth": 8192,
    "accumulator_depth": 2048,
    "simd_registers_depth": 1,
    "stride0_depth": 8,
    "stride1_depth": 8,
    "number_of_threads": 1,
    "thread_queue_depth": 8
}
result_names_list = ["Model",
                     "Data type",
                     "Array size",
                     "DRAM0 memory size",
                     "DRAM1 memory size",
                     "Local memory size",
                     "Accumulator memory size",
                     "Stride #0 size",
                     "Stride #1 size",
                     "Operand #0 size",
                     "Operand #1 size",
                     "Operand #2 size",
                     "Instruction size",
                     "DRAM0 maximum usage",
                     "DRAM0 aggregate usage",
                     "DRAM1 maximum usage",
                     "DRAM1 aggregate usage",
                     "Local memory maximum usage",
                     "Local memory aggregate usage",
                     "Accumumator memory maximum usage",
                     "Accumumator memory aggregate usage",
                     "Number of layers",
                     "Maximum number of stages",
                     "Maximum number of partitions",
                     "Execution latency (MCycles)",
                     "Aggregate latency (MCycles)",
                     "Execution energy (MUnits)",
                     "Aggregate energy (MUnits)",
                     "MAC efficiency (%)",
                     "Total number of instructions",
                     "Compilation time",
                     "True consts scalar",
                     "Consts utilization",
                     "True MACs",
                     "MAC efficiency"]
result_dict = {
    "Model": "resnet20",
    "Data type": "FP16BP8",
    "Array size": 8,
    "DRAM0 memory size": 8,
    "DRAM1 memory size": 8,
    "Local memory size": 8,
    "Accumulator memory size": 8,
    "Stride #0 size": 8,
    "Stride #1 size": 8,
    "Operand #0 size": 8,
    "Operand #1 size": 8,
    "Operand #2 size": 8,
    "Instruction size": 8,
    "DRAM0 maximum usage": 8,
    "DRAM0 aggregate usage": 8,
    "DRAM1 maximum usage": 8,
    "DRAM1 aggregate usage": 8,
    "Local memory maximum usage": 8,
    "Local memory aggregate usage": 8,
    "Accumumator memory maximum usage": 8,
    "Accumumator memory aggregate usage": 8,
    "Number of layers": 8,
    "Maximum number of stages": 8,
    "Maximum number of partitions": 8,
    "Execution latency (MCycles)": 8,
    "Aggregate latency (MCycles)": 8,
    "Execution energy (MUnits)": 8,
    "Aggregate energy (MUnits)": 8,
    "MAC efficiency (%)": 8,
    "Total number of instructions": 8,
    "Compilation time": 8,
    "True consts scalar": 8,
    "Consts utilization": 8,
    "True MACs": 8,
    "MAC efficiency": 8
}
special_name_list_bits = ["DRAM0 memory size", "DRAM1 memory size", "Local memory size", "Accumulator memory size"]
special_name_list_2 = ["DRAM0 maximum usage", "DRAM0 aggregate usage", "DRAM1 maximum usage", "DRAM1 aggregate usage",
                       "Local memory aggregate usage", "Local memory maximum usage", "Accumumator memory maximum usage",
                       "Accumumator memory aggregate usage"]
result_dict_temp = {
    "model_name": "resnet20",
    "data_type": "FP16BP8",
    "Array_Size": 8,

    "DRAM0_Size_Vectors": 4194304,
    "DRAM0_Size_scalars": 4194304,
    "DRAM0_Size_bits": 4194304,

    "DRAM1_Size_Vectors": 4194304,
    "DRAM1_Size_scalars": 4194304,
    "DRAM1_Size_bits": 4194304,

    "Local_Memory_Size_Vectors": 8192,
    "Local_Memory_Size_Scalars": 8192,
    "Local_Memory_Size_Bits": 8192,

    "Accumulator_Memory_Size_Vectors": 8192,
    "Accumulator_Memory_Size_Scalars": 8192,
    "Accumulator_Memory_Size_Bits": 8192,

    "Stride_0_bits": 3,
    "Stride_1_bits": 3,
    "Operand_0_bits": 3,
    "Operand_1_bits": 3,
    "Operand_2_bits": 3,

    "Instruction_Size_bits": 9,

    "DRAM0_Maximum_Vectors": 26624,
    "DRAM0_Maximum_Scalars": 26624,

    "DRAM1_Maximum_Vectors": 26624,
    "DRAM1_Maximum_Scalars": 26624,

    "DRAM0_Aggregate_Vectors": 26624,
    "DRAM0_Aggregate_Scalars": 26624,

    "DRAM1_Aggregate_Vectors": 26624,
    "DRAM1_Aggregate_Scalars": 26624,

    "Local_memory_maximum_vectors": 8192,
    "Local_memory_maximum_scalars": 8192,

    "Local_memory_aggregate_vectors": 8192,
    "Local_memory_aggregate_scalars": 8192,

    "Accumumator_memory_maximum_vectors": 8192,
    "Accumumator_memory_maximum_scalars": 8192,

    "Accumumator_memory_aggregate_vectors": 8192,
    "Accumumator_memory_aggregate_scalars": 8192,

    "NumberOfLayer": 27,

    "MaximumNumberOfStage": 16,
    "MaximumNumberPartition": 24,

    "ExecutionLatency": 1.916,
    "AggregateLatency": 1.916,
    "ExecutionEnergy": 100000,
    "AggregateEnergy": 100000,
    "MACEfficiency": 100,
    "TotalNumberOfInstructions": 0,
    "CompilationTime": 0,
    "TrueConstsScalarSize": 568,
    "ConstsUtilization": 0,
    "TrueMACS": 568,
    "MACEfficiency2": 0
}

def move_file(compiled_model_name, output_path):
    """
    Move tmodel, tprog and tdata to the specified directory
    Args :
        - compiled_model_name (str) : *_onnx_{arch}, correspond to output of tensil
    """
    print("Moving file")

    print(os.getcwd())
    print(compiled_model_name)

    compiled_model_name = compiled_model_name.replace("-", "_")
    print(compiled_model_name)
    # Moving Compiled model
    try :
        os.rename(compiled_model_name + ".tmodel", output_path + compiled_model_name + ".tmodel")
    except :
        print("No tmodel file")
    try :
        os.rename(compiled_model_name + ".tprog", output_path + compiled_model_name + ".tprog")
    except :
        print("No tprog file")
    try :
        os.rename(compiled_model_name + ".tdata", output_path + compiled_model_name + ".tdata")
    except :
        print("No data file")


def save_compilation_result(logs, name, path):
    """
    Save the logs in a csv file
    """
    print("logs in:", path+name+".txt")


    with open(path+name+".txt","wb") as file:

        file.write(logs)

    # Logs Data Collecting
    lines = logs.splitlines()
    for line in lines:
        temp = line.decode()
        for namee in result_names_list:
            if namee in temp:
                tList = temp.split()
                result_dict[namee] = tList[1:]

    # Result Save in csv
    with open(path + name + ".csv", "w+") as saveFile:
        w = csv.writer(saveFile)
        for key, val in result_dict.items():
            # write every key and value to file
            if key in special_name_list_2:
                w.writerow([key, val[-2:]])
            elif key in special_name_list_bits:
                w.writerow([key, val[-3:]])
            else:
                if type(key) == type([]):  # type == list
                    w.writerow([key, val[-1]])
                else:
                    w.writerow([key, val])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-path', type=str, help='path to onnx file')
    parser.add_argument('--arch-path', type=str, default= "arch/custom_perf.tarch",help='path to tensil architecture file')
    parser.add_argument('--output-dir', type=str, default= "tensil/",help='path to script output directory')
    parser.add_argument('--onnx-output', type=str, default= "Output",help='name of the onnx output layer')
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Network Compilation
    print("Tensil compiling...")

    network = args.onnx_path
    pwd = os.getcwd()
    client = docker.from_env()
    name_net = network.split(sep="/")
    name_net = name_net[-1][:-5]
    try:
        # - a : architecture
        # - m : onnx model
        # -v  : verbose

        # additional summary (all default to false):
        # -s : print summary
        # --layers-summary
        # --scheduler-summary
        # --partition-summary
        # --strides-summary
        # --instructions-summary

        summary_flags=["-s", "true","--layers-summary","true","--scheduler-summary","true","--partitions-summary","true","--strides-summary","true","--instructions-summary","true"]

        log_casa = client.containers.run("tensilai/tensil:latest",
                                            ["tensil", "compile", "-a", args.arch_path, "-m", network,
                                            "-o", args.onnx_output, "-t", args.output_dir]+summary_flags,
                                            volumes=[pwd + ":/work"],
                                            working_dir="/work",
                                            stderr=True)

        save_compilation_result(log_casa, name_net, args.output_dir)
        print("-------------------------")
        print("-------------------------")
        print("-------------------------")
        print("Compilation successful !!")
        print("-------------------------")
        print("-------------------------")

    except docker.errors.ContainerError as exc:
        with open(args.output_dir + name_net + ".txt","wb") as file:

            file.write(exc.container.logs())
        print("-------------------------")
        print("-------------------------")
        print("-------------------------")
        print("Compilation unsuccessful")
        print("error was: ")
        print("------------------------")
        print(exc.container.logs())
        print("------------------------")

if __name__ == "__main__":
    main()
#!/bin/bash

set -e
set -u
set -o pipefail

export PATH=/data/fengdahu/npusim/release/cmake-3.31.3-linux-x86_64/bin:$PATH

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR_ABS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

LLM_TEST_DIR_ABS="${SCRIPT_DIR_ABS}"
BUILD_DIR_ABS=$(cd "${SCRIPT_DIR_ABS}/../../build" && pwd)

NPUSIM_STDOUT_TMP_BASENAME="npusim_output.tmp.$$"
TEMP_JSON_CONFIG_BASENAME="temp_config.json"
BUILD_LOG_BASENAME="build.log"

NPUSIM_STDOUT_TMP_FULL_PATH="${BUILD_DIR_ABS}/${NPUSIM_STDOUT_TMP_BASENAME}"
TEMP_JSON_CONFIG_FULL_PATH="${LLM_TEST_DIR_ABS}/${TEMP_JSON_CONFIG_BASENAME}"
BUILD_LOG_FULL_PATH="${BUILD_DIR_ABS}/${BUILD_LOG_BASENAME}"

cleanup() {
    echo "INFO: Cleaning up temporary files..."
    rm -f "${NPUSIM_STDOUT_TMP_FULL_PATH}"
    rm -f "${TEMP_JSON_CONFIG_FULL_PATH}"
    rm -f "${BUILD_LOG_FULL_PATH}"
}

# trap cleanup EXIT INT TERM

if [ "$#" -ne 1 ]; then
    echo "Usage: ./${SCRIPT_NAME} <test_batch_file>"
    exit 1
fi

TEST_BATCH_FILE="$1"
OUTPUT_BATCH_FILE="${LLM_TEST_DIR_ABS}/${TEST_BATCH_FILE}.results"

if [ ! -f "${TEST_BATCH_FILE}" ]; then
    echo "ERROR: Test batch file '${TEST_BATCH_FILE}' not found."
    exit 1
fi

LINE_NUM=0
while IFS= read -r line || [[ -n "$line" ]]; do
    LINE_NUM=$((LINE_NUM + 1))
    echo ""
    echo "INFO: Processing line ${LINE_NUM}: ${line}"

    FIELD_COUNT=$(echo "$line" | awk '{print NF}')

    ORIGINAL_PWD=$(pwd)
    cd "${BUILD_DIR_ABS}"

    if [ "$FIELD_COUNT" -eq 10 ]; then
        echo "INFO: Detected 10 fields, using legacy mode."
        read -r NPUSIM_WORKLOAD_CONFIG JSON_X JSON_CORE_ID JSON_EXU_X JSON_EXU_Y JSON_SFU_X JSON_SRAM_BITWIDTH JSON_SRAM_MAX_SIZE JSON_COMM_PAYLOAD JSON_DRAM_BANDWIDTH<<<"$line"

        CONFIG_CONTENT=$(
            cat <<EOF
{
    "x": ${JSON_X},
    "comm_acc": ${JSON_COMM_PAYLOAD},
    "cores": [
        {
            "id": ${JSON_CORE_ID},
            "exu_x": ${JSON_EXU_X},
            "exu_y": ${JSON_EXU_Y},
            "sfu_x": ${JSON_SFU_X},
            "sram_bitwidth": ${JSON_SRAM_BITWIDTH}
        }
    ]
}
EOF
        )

        echo "$CONFIG_CONTENT" >"${TEMP_JSON_CONFIG_FULL_PATH}"
        echo "INFO: Created temp JSON config: ${TEMP_JSON_CONFIG_FULL_PATH}"
        echo "INFO: Executing: ./npusim --workload-config=\"../llm/test/workload_config/${NPUSIM_WORKLOAD_CONFIG}\" --hardware-config=\"../llm/test/${TEMP_JSON_CONFIG_BASENAME}\" --sram-max=\"${JSON_SRAM_MAX_SIZE}\" --df_dram_bw=\"${JSON_DRAM_BANDWIDTH}\""

        ./npusim --workload-config="../llm/test/workload_config/${NPUSIM_WORKLOAD_CONFIG}" \
                 --hardware-config="../llm/test/${TEMP_JSON_CONFIG_BASENAME}" \
                 --sram-max="${JSON_SRAM_MAX_SIZE}" \
                 --df_dram_bw="${JSON_DRAM_BANDWIDTH}" \
                 > "${NPUSIM_STDOUT_TMP_BASENAME}"

    elif [ "$FIELD_COUNT" -eq 2 ]; then
        echo "INFO: Detected 2 fields, using new direct mode."
        read -r WORKLOAD_CONFIG HARDWARE_CONFIG <<<"$line"

        ./npusim --workload-config="../llm/test/workload_config/${WORKLOAD_CONFIG}" \
                 --hardware-config="../llm/test/${HARDWARE_CONFIG}" \
                 --df_dram_bw 32 \
                 > "${NPUSIM_STDOUT_TMP_BASENAME}"
    else
        echo "ERROR: Line ${LINE_NUM} has invalid number of fields ${FIELD_COUNT} (expected 2 or 10)."
        echo -e "${line}\tERROR: Invalid parameter count" >>"${OUTPUT_BATCH_FILE}"
        cd "${ORIGINAL_PWD}"
        continue
    fi

    RESULT_STRING=""
    RECORD_FOUND=0
    while IFS= read -r line_catch; do
        if [[ "$line_catch" == "[CATCH TEST]"* ]]; then
            remaining_content="${line_catch#"[CATCH TEST]"}"
            remaining_content="${remaining_content##+([ \t])}"
            RESULT_STRING="${remaining_content}"
            RECORD_FOUND=1
            break
        fi
        if [[ "$line_catch" == "[ERROR]"* ]]; then
            remaining_content="${line_catch#"[ERROR]"}"
            remaining_content="${remaining_content##+([ \t])}"
            RESULT_STRING="${remaining_content}"
            RECORD_FOUND=1
            break
        fi
    done < <(tac "${NPUSIM_STDOUT_TMP_BASENAME}")

    if [ $RECORD_FOUND -eq 0 ]; then
        echo "WARNING: No [CATCH TEST] string found for line ${LINE_NUM}."
    fi

    echo -e "${line}\t${RESULT_STRING}" >>"${OUTPUT_BATCH_FILE}"
    echo -e "${line}\t${RESULT_STRING}"
    cd "${ORIGINAL_PWD}"

done <"${TEST_BATCH_FILE}"

echo ""
echo "INFO: Batch processing complete. Results are in ${OUTPUT_BATCH_FILE}"
exit 0

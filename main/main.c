#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "esp_afe_sr_iface.h"
#include "esp_afe_sr_models.h"

static const char *TAG = "WAKE";

/* 测试唤醒词时设为 0，关闭二进制帧输出；正常使用时设为 1 */
#define BINARY_OUTPUT   0

/* 单麦模式：仅 I2S0，不初始化 I2S1。设为 1 启用 */
#define SINGLE_MIC      1

#define SAMPLE_RATE     16000
#define DMA_FRAME_MS    32
#define SAMPLES_PER_CH  512
#define DMA_BUF_LEN     (SAMPLES_PER_CH * 2)
#define CHANNEL_COUNT   3

#define I2S0_PIN_SD     GPIO_NUM_40
#define I2S1_PIN_SD     GPIO_NUM_39
#define I2S_PIN_SCK     GPIO_NUM_41
#define I2S_PIN_WS      GPIO_NUM_42

/* 帧结构: sync(2) + frm(4) + ts(8) + nch(1) + nspc(2) + data(CH*SPC*2) + crc(2) */
#define SYNC1           0xAA
#define SYNC2           0x55
#define FRAME_DATA_BYTES (CHANNEL_COUNT * SAMPLES_PER_CH * 2)
#define FRAME_SIZE      (2 + 4 + 8 + 1 + 2 + FRAME_DATA_BYTES + 2)

static int16_t i2s0_read_buf[DMA_BUF_LEN];
static int16_t i2s1_read_buf[DMA_BUF_LEN];
static int16_t mic_left[SAMPLES_PER_CH];
static int16_t mic_right[SAMPLES_PER_CH];
static int16_t mic_third[SAMPLES_PER_CH];
static int16_t dummy[SAMPLES_PER_CH];

static i2s_chan_handle_t rx_handle0 = NULL;
static i2s_chan_handle_t rx_handle1 = NULL;
static uint8_t frame[FRAME_SIZE];

/* AFE */
static esp_afe_sr_iface_t *afe_handle = NULL;
static esp_afe_sr_data_t *afe_data = NULL;
static int16_t *afe_feed_buf = NULL;
static int afe_chunksize = 0;
static int afe_buf_idx = 0;

/* ---------- CRC16 ---------- */
static uint16_t crc16_update(uint16_t crc, uint8_t data)
{
    crc ^= (uint16_t)data << 8;
    for (int i = 0; i < 8; i++)
        crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1);
    return crc;
}

static uint16_t crc16_buf(const uint8_t *buf, size_t len)
{
    uint16_t crc = 0;
    for (size_t i = 0; i < len; i++)
        crc = crc16_update(crc, buf[i]);
    return crc;
}

/* ---------- I2S ---------- */
static bool i2s0_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_PIN_SCK,
            .ws   = I2S_PIN_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S0_PIN_SD,
            .invert_flags = {
                .mclk_inv = false, .bclk_inv = false, .ws_inv = false,
            },
        },
    };
    esp_err_t err = i2s_new_channel(&chan_cfg, NULL, &rx_handle0);
    if (err != ESP_OK) { printf("# FAIL i2s0 new: %s\n", esp_err_to_name(err)); return false; }
    err = i2s_channel_init_std_mode(rx_handle0, &std_cfg);
    if (err != ESP_OK) { printf("# FAIL i2s0 std: %s\n", esp_err_to_name(err)); return false; }
    err = i2s_channel_enable(rx_handle0);
    if (err != ESP_OK) { printf("# FAIL i2s0 en: %s\n", esp_err_to_name(err)); return false; }
    printf("# OK: i2s0 MASTER (GPIO40 SD, GPIO41 SCK, GPIO42 WS)\n");
    return true;
}

static bool i2s1_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_SLAVE);
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_PIN_SCK,
            .ws   = I2S_PIN_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S1_PIN_SD,
            .invert_flags = {
                .mclk_inv = false, .bclk_inv = false, .ws_inv = false,
            },
        },
    };
    esp_err_t err = i2s_new_channel(&chan_cfg, NULL, &rx_handle1);
    if (err != ESP_OK) { printf("# FAIL i2s1 new: %s\n", esp_err_to_name(err)); return false; }
    err = i2s_channel_init_std_mode(rx_handle1, &std_cfg);
    if (err != ESP_OK) { printf("# FAIL i2s1 std: %s\n", esp_err_to_name(err)); return false; }
    err = i2s_channel_enable(rx_handle1);
    if (err != ESP_OK) { printf("# FAIL i2s1 en: %s\n", esp_err_to_name(err)); return false; }
    printf("# OK: i2s1 SLAVE (GPIO39 SD)\n");
    return true;
}

static void deinterleave(const int16_t *buf, int16_t *left, int16_t *right, int pairs)
{
    for (int i = 0; i < pairs; i++) {
        left[i]  = buf[i * 2];
        right[i] = buf[i * 2 + 1];
    }
}

static void build_frame(uint8_t *frame, uint32_t fc, int64_t ts_us,
                        const int16_t *L, const int16_t *R, const int16_t *C, int n)
{
    int p = 0;
    frame[p++] = SYNC1;
    frame[p++] = SYNC2;

    frame[p++] = (fc >>  0) & 0xFF;
    frame[p++] = (fc >>  8) & 0xFF;
    frame[p++] = (fc >> 16) & 0xFF;
    frame[p++] = (fc >> 24) & 0xFF;

    for (int i = 0; i < 8; i++)
        frame[p++] = (ts_us >> (i * 8)) & 0xFF;

    frame[p++] = CHANNEL_COUNT;
    frame[p++] = (n >> 0) & 0xFF;
    frame[p++] = (n >> 8) & 0xFF;

    for (int ch = 0; ch < CHANNEL_COUNT; ch++) {
        const int16_t *src = (ch == 0) ? L : ((ch == 1) ? R : C);
        for (int i = 0; i < n; i++) {
            int16_t v = src[i];
            frame[p++] = (v >> 0) & 0xFF;
            frame[p++] = (v >> 8) & 0xFF;
        }
    }

    uint16_t crc = crc16_buf(frame + 2, p - 2);
    frame[p++] = (crc >> 0) & 0xFF;
    frame[p++] = (crc >> 8) & 0xFF;
}

/* ---------- AFE ---------- */
static bool afe_init(void)
{
    srmodel_list_t *models = esp_srmodel_init("model");
    if (models == NULL) {
        ESP_LOGE(TAG, "esp_srmodel_init failed");
        return false;
    }

    afe_config_t *cfg = afe_config_init("M", models, AFE_TYPE_SR, AFE_MODE_HIGH_PERF);
    if (cfg == NULL) {
        ESP_LOGE(TAG, "afe_config_init failed");
        return false;
    }

    cfg->aec_init = false;
    cfg->afe_linear_gain = 10.0f;  /* INMP441 增益补偿 */

    afe_handle = esp_afe_handle_from_config(cfg);
    if (afe_handle == NULL) {
        ESP_LOGE(TAG, "esp_afe_handle_from_config failed");
        return false;
    }

    afe_data = afe_handle->create_from_config(cfg);
    if (afe_data == NULL) {
        ESP_LOGE(TAG, "create_from_config failed");
        return false;
    }

    afe_chunksize = afe_handle->get_feed_chunksize(afe_data);
    ESP_LOGI(TAG, "AFE ready, feed_chunksize=%d", afe_chunksize);

    afe_feed_buf = (int16_t *)malloc(afe_chunksize * sizeof(int16_t));
    assert(afe_feed_buf);
    afe_buf_idx = 0;

    return true;
}

void app_main(void)
{
    printf("# HELLO 3-mic INMP441 + WakeNet\n");
    fflush(stdout);

    if (!i2s0_init()) goto fail;
#if !SINGLE_MIC
    if (!i2s1_init()) goto fail;
#endif

    printf("# sample_rate=%d\n", SAMPLE_RATE);
    printf("# frame_size_bytes=%d\n", FRAME_SIZE);
    printf("# channels=%d\n", CHANNEL_COUNT);
    printf("# samples_per_channel=%d\n", SAMPLES_PER_CH);
    printf("# mic_distance_m=0.20\n");
    fflush(stdout);

    if (!afe_init()) {
        printf("# WARN: AFE init failed, continuing without wake word\n");
        fflush(stdout);
    } else {
        printf("# OK: AFE ready\n");
        fflush(stdout);
    }

    size_t bytes0 = 0, bytes1 = 0;
    uint32_t fc = 0;
    uint32_t timeout = 0;
    int64_t t0 = esp_timer_get_time();

    while (true) {
        esp_err_t r0 = i2s_channel_read(rx_handle0, i2s0_read_buf, sizeof(i2s0_read_buf),
                                         &bytes0, pdMS_TO_TICKS(500));
#if SINGLE_MIC
        esp_err_t r1 = ESP_OK;
        bytes1 = bytes0;
        memset(i2s1_read_buf, 0, sizeof(i2s1_read_buf));
#else
        esp_err_t r1 = i2s_channel_read(rx_handle1, i2s1_read_buf, sizeof(i2s1_read_buf),
                                         &bytes1, pdMS_TO_TICKS(500));
#endif

        /* 诊断：每次 I2S 读取结果 */
        {
            static uint32_t diag_cnt = 0;
            if (++diag_cnt % 100 == 1) {
                printf("# diag: r0=%d r1=%d bytes0=%u bytes1=%u\n",
                       (int)r0, (int)r1, (unsigned)bytes0, (unsigned)bytes1);
                fflush(stdout);
            }
        }

        if (r0 == ESP_ERR_TIMEOUT || r1 == ESP_ERR_TIMEOUT) {
            timeout++;
            if (timeout % 10 == 1) {
                printf("# timeout x%lu (%.1fs) r0=%s r1=%s\n",
                       (unsigned long)timeout,
                       (double)(esp_timer_get_time() - t0) / 1e6,
                       r0 == ESP_ERR_TIMEOUT ? "TO" : "OK",
                       r1 == ESP_ERR_TIMEOUT ? "TO" : "OK");
                fflush(stdout);
            }
            continue;
        }
        if (r0 != ESP_OK || r1 != ESP_OK) continue;
        if (bytes0 == 0 || bytes1 == 0) continue;

        int pairs = bytes0 / sizeof(int16_t) / 2;
        if (pairs > SAMPLES_PER_CH) pairs = SAMPLES_PER_CH;

        deinterleave(i2s0_read_buf, mic_left, mic_right, pairs);
        deinterleave(i2s1_read_buf, mic_third, dummy, pairs);

        /* 每秒确认一次数据流通 */
        {
            static int64_t last_alive_ts = 0;
            int64_t now = esp_timer_get_time();
            if (now - last_alive_ts > 1000000) {
                printf("# I2S OK: fc=%lu, mic_left[0]=%d\n",
                       (unsigned long)fc, (int)mic_left[0]);
                fflush(stdout);
                last_alive_ts = now;
            }
        }

        /* 喂 AFE - 渐进式累积，每次填满 afe_chunksize 就 feed */
        if (afe_data && afe_feed_buf) {
            for (int i = 0; i < pairs; i++) {
                afe_feed_buf[afe_buf_idx++] = mic_left[i];
                if (afe_buf_idx >= afe_chunksize) {
                    afe_handle->feed(afe_data, afe_feed_buf);
                    afe_buf_idx = 0;

                    /* 每次 feed 后立即 fetch 以排空输出缓冲区 */
                    afe_fetch_result_t *res;
                    while ((res = afe_handle->fetch_with_delay(afe_data, 0)) != NULL) {
                        if (res->wakeup_state == WAKENET_DETECTED) {
                            printf("\n# >>> WAKE WORD DETECTED <<<\n\n");
                            fflush(stdout);
                        }
                    }
                }
            }
        }

        int64_t ts = esp_timer_get_time() - t0;
        fc++;

#if BINARY_OUTPUT
        build_frame(frame, fc, ts, mic_left, mic_right, mic_third, pairs);
        fwrite(frame, 1, FRAME_SIZE, stdout);
        fflush(stdout);
#endif
        vTaskDelay(pdMS_TO_TICKS(1));
    }

fail:
    printf("# FATAL: init failed\n");
    while (1) vTaskDelay(pdMS_TO_TICKS(1000));
}

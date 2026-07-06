#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_rom_uart.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"

#define SAMPLE_RATE     16000
#define DMA_FRAME_MS    32
#define SAMPLES_PER_CH  512
#define UART_BAUD        921600
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

void app_main(void)
{
    esp_rom_uart_set_clock_baudrate(0, UART_BAUD);
    printf("# HELLO 3-mic INMP441 (binary protocol)\n");
    fflush(stdout);

    if (!i2s0_init()) goto fail;
    if (!i2s1_init()) goto fail;

    printf("# sample_rate=%d\n", SAMPLE_RATE);
    printf("# frame_size_bytes=%d\n", FRAME_SIZE);
    printf("# channels=%d\n", CHANNEL_COUNT);
    printf("# samples_per_channel=%d\n", SAMPLES_PER_CH);
    printf("# mic_distance_m=0.20\n");
    printf("# OK: I2S ready\n");
    fflush(stdout);

    size_t bytes0 = 0, bytes1 = 0;
    uint32_t fc = 0;
    uint32_t timeout = 0;
    int64_t t0 = esp_timer_get_time();

    while (true) {
        esp_err_t r0 = i2s_channel_read(rx_handle0, i2s0_read_buf, sizeof(i2s0_read_buf),
                                         &bytes0, pdMS_TO_TICKS(500));
        esp_err_t r1 = i2s_channel_read(rx_handle1, i2s1_read_buf, sizeof(i2s1_read_buf),
                                         &bytes1, pdMS_TO_TICKS(500));

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

        int64_t ts = esp_timer_get_time() - t0;
        fc++;

        build_frame(frame, fc, ts, mic_left, mic_right, mic_third, pairs);
        fwrite(frame, 1, FRAME_SIZE, stdout);
        fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(1));
    }

fail:
    printf("# FATAL: init failed\n");
    while (1) vTaskDelay(pdMS_TO_TICKS(1000));
}

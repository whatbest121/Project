#include <Arduino_FreeRTOS.h> #include <queue.h>
struct pinRead
{
    int pin;
    int value;
};
QueueHandle_t structQueue;
void setup()
{
    structQueue = xQueueCreate(10, // Queue length sizeof(struct pinRead) // Queue item size
    );
    if (structQueue != NULL)
    {
        // Create task that consumes the queue if it was created.
        xTaskCreate(TaskSerial, // Task function "Serial", // A name just for humans
                    128,        // This stack size can be checked & adjusted by reading the Stack Highwater
                    NULL,
                    2, // Priority, with 3 (configMAX_PRIORITIES - 1) being the highest, and 0 being the lowest. NULL);
                    // Create task that publish data in the queue if it was created. xTaskCreate(TaskAnalogReadPin0, // Task function
                    "AnalogReadPin0", // Task name 64, // Stack size
                    NULL,
                    1, // Priority
                    NULL);
        // Create other task that publish data in the queue if it was created. xTaskCreate(TaskAnalogReadPin1, // Task function
"AnalogReadPin1", // Task name 64, // Stack size
NULL,
1, // Priority
NULL);
// Create other task that publish data in the queue if it was created. xTaskCreate(TaskAnalogReadPin2, // Task function
"AnalogReadPin1", // Task name 64, // Stack size
NULL,
1, // Priority
NULL);
// Create other task that publish data in the queue if it was created. xTaskCreate(TaskAnalogReadPin3, // Task function
"AnalogReadPin1", // Task name 64, // Stack size
NULL,
1, // Priority
NULL);
// Create other task that publish data in the queue if it was created. xTaskCreate(TaskAnalogReadPin4, // Task function
"AnalogReadPin1", // Task name 64, // Stack size
NULL,
1, // Priority
NULL);
// Create other task that publish data in the queue if it was created. xTaskCreate(TaskAnalogReadPin5, // Task function
"AnalogReadPin5", // Task name 64, // Stack size
NULL,
1, // Priority
NULL);
    }
}

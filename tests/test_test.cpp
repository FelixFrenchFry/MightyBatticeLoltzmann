//#include "simulation.h"
#include <gtest/gtest.h>

// Demonstrate some basic assertions.
TEST(TestTest, BasicAssertions) {
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 12, 84);
    // Testing if we can call a function from our MD library
    //HelloWorld();
}

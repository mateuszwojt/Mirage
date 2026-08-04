#include <chrono>

namespace Mirage
{

    class Timer
    {
    private:
        // Type aliases to make accessing nested type easier
        using clock_t = std::chrono::steady_clock;
        using millisecond_t = std::chrono::duration<double, std::milli>;

        std::chrono::time_point<clock_t> m_startTime;

    public:
        Timer() : m_startTime(clock_t::now())
        {
        }

        void reset()
        {
            m_startTime = clock_t::now();
        }

        double elapsed() const
        {
            return std::chrono::duration_cast<millisecond_t>(clock_t::now() - m_startTime).count();
        }
    };

}

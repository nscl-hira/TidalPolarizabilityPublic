#include <tuple>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <boost/python/numpy.hpp>
#include <boost/scoped_array.hpp>
#include <boost/numeric/odeint.hpp>
#include "spline.h"

typedef std::vector<double> state_type;
typedef boost::numeric::odeint::runge_kutta_cash_karp54<state_type> error_stepper_type;
typedef tk::spline SEOS;

const double MEVFM3 = 1.60217646e32;    // in J/m3 --> a converter from (MeV/fm3)
const double TOPA = 4.4173085e36;    // Pressure units in Pascal
const double TOKM = 1.47671618;        // Kilometers
const double TOJM3 = 4.4173085e36;    // Joules/m3
const double G = 6.67408e-11;        // gravitational constant
const double MODOT = 1.989e30;        // Mass of sun in kg
const double C = 299792458.;        // speed of light in m/s

SEOS EOSFromFile(const std::string& t_filename)
{
    std::ifstream file(t_filename.c_str());
    assert(file.is_open());
    
    // get rid of the header (which is 4 row deep)
    std::string line;
    for(int i = 0; i < 4; ++i) std::getline(file, line);

    // only get the first 2 column
    std::vector<double> energy_density, pressure;
    bool first_line = true;
    while(std::getline(file, line))
    {
        std::stringstream ss(line);
        double edensity, pres, temp;
        if(!(ss >> edensity >> pres >> temp >> temp))
        {
            std::cerr << " Cannot read line " << line << "\n";
            continue;
        }
    if(!first_line)
        {
        if(pres*MEVFM3/TOPA > pressure.back())
        {
                energy_density.push_back(edensity*MEVFM3/TOJM3);
                pressure.push_back(pres*MEVFM3/TOPA);
        }
        }
        else
        {
            energy_density.push_back(edensity*MEVFM3/TOJM3);
            pressure.push_back(pres*MEVFM3/TOPA);
            first_line = false;
        }
        
    }

    SEOS s;
    s.set_points(pressure, energy_density);
    return s;
}

class MaxCompact
{
public:
    MaxCompact(double t_epsilon) : epsilon_(t_epsilon) {};
    MaxCompact(const MaxCompact& t_maxcompact) : epsilon_(t_maxcompact.epsilon_) {};
    
    double operator() (double t_pressure)
    {
        return t_pressure + epsilon_;    
    };

    double deriv(int order, double t_pressure)
    {
        return 1.;
    };
private:
    double epsilon_;
};

template<class EOS>
class TOV_eq
{
private:
    // EOS need to be able to map pressure to energy
    EOS eos_;
public:
    TOV_eq(const EOS& t_eos) : eos_(t_eos) {};
    
    void operator() ( const state_type &x, state_type &dxdt, const double r)
    {
        double P = x[0];
        double M = x[1];
        double E = eos_(P);
        //std::cout << "Energy " << E*TOJM3/MEVFM3 << " Pressure " << P*TOPA/MEVFM3 << "\r";
        dxdt[0] = -(E + P)*(M + r*r*r*P)/(r*(r - 2.*M));
        dxdt[1] = r*r*E;

        // this notation folloes PhysRevC.87.015806
        // the part above this line solves TOV equation
        // the part below calculate tidal Love number k2
        double F_r = (r - r*r*r*(E - P))/(r - 2*M);
        double Q_r_part1 = r*(5*E + 9*P + (E + P)*eos_.deriv(1, P) - 6./(r*r))/(r - 2*M);
        double Q_r_part2 = (M + r*r*r*P)/(r*r*(1 - 2*M/r));
        double Q_r = Q_r_part1 - 4*Q_r_part2*Q_r_part2;
    
        double y = x[2];
        dxdt[2] = - (y*y + y*F_r +r*r*Q_r)/r;    

        if(P < 0)
            dxdt[0] = dxdt[1] = dxdt[2] = 0;
    };

    double GetK2(const state_type &x, const double R)
    {
        double M = x[1];
        double yR = x[2];
        double Rs = 2*M;
        double Rs_R = Rs/R;
        double Rs_R2 = Rs_R*Rs_R;

        double part1 = 1./20.*pow(Rs_R, 5)*pow(1 - Rs_R, 2)*(2 - yR + (yR - 1)*Rs_R);
        double part2 = Rs_R*(6. -3.*yR + 3.*Rs*(5*yR - 8.)/(2.*R));
        double part3 = 1./4.*Rs_R*Rs_R2*(26. - 22.*yR + (Rs_R)*(3*yR - 2) + Rs_R2*(1 + yR));
        double part4 = 3*(1 - Rs_R)*(1 - Rs_R)*(2 - yR + (yR - 1)*Rs_R)*log(1 - Rs_R);
        return part1/(part2 + part3 + part4);
    }
};

struct CheckpointState
{
public:
    CheckpointState(const std::vector<double>& t_checkpoint_pressure, 
                    std::vector<double>& t_mass, 
                    std::vector<double>& t_radius,
                    double &t_R,
                    bool t_verbose = false) : checkpoint_pressure_(t_checkpoint_pressure),
                                              checkpoint_index(0),
                                              mass_(t_mass),
                                              radius_(t_radius),
                                              R(t_R),
                                              verbose_(t_verbose) {};

    void operator()( const state_type &state, double r)
    {
        double P = state[0]/MEVFM3*TOPA;
        R = r*TOKM;

        if(verbose_)
            std::cout << "r (km): " 
                      << std::setw(10) << R << " P (Pa): " 
                      << std::setw(10) << P << " M: " 
                      << std::setw(10) << state[1] << " y: " 
                      << std::setw(10) << state[2] << "\n";
        if(P < 1e-15)
            throw std::invalid_argument("Pressure is now negative");

        if(checkpoint_index < checkpoint_pressure_.size())
            if(P < checkpoint_pressure_[checkpoint_index])
            {
                mass_.push_back(state[1]);
                radius_.push_back(R);
                ++checkpoint_index;
            }
    }

private:
    std::vector<double> checkpoint_pressure_;
    int checkpoint_index;
    std::vector<double>& mass_;
    std::vector<double>& radius_;
    double& R;
    bool verbose_;
};

struct CheckpointSaveAll
{
public:
    CheckpointSaveAll(std::vector<double>& t_mass, std::vector<double>& t_radius, std::vector<double>& t_pressure) :
        mass_(t_mass), radius_(t_radius), pressure_(t_pressure){};

    void operator()(const state_type &state, double r)
    {
        double P = state[0]/MEVFM3*TOPA;
        double R = r*TOKM;

        mass_.push_back(state[1]);
        pressure_.push_back(P);
        radius_.push_back(R);

        if(P < 1e-10)
            throw std::invalid_argument("Pressure is now negative");
    }
private:
    std::vector<double>& mass_;
    std::vector<double>& radius_;
    std::vector<double>& pressure_;
};

typedef std::vector<double> list;

// wrappers needed to convert numpy to and from python
namespace p = boost::python;
namespace np = boost::python::numpy;

list wrap_from_ndarray(np::ndarray const & array)
{
    if (array.get_dtype() != np::dtype::get_builtin<double>()) 
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        p::throw_error_already_set();
    }
    if (array.get_nd() != 1) 
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
        p::throw_error_already_set();
    }

    list result;
    int rows = array.shape(0);
    for(int i = 0; i < rows; ++i)
        result.push_back(*(reinterpret_cast<double*>(array.get_data()) + i));
 
    return result;
}

np::ndarray wrap_to_ndarray(const list& t_list)
{
    Py_intptr_t shape[1] = {t_list.size() };
    np::ndarray result = np::zeros(1, shape, np::dtype::get_builtin<double>());
    std::copy(t_list.begin(), t_list.end(), reinterpret_cast<double*>(result.get_data()));
    return result;
}

std::tuple<double, double, double, list, list> TidalLove_individual(const std::string& t_EOS_filename,
                                                                    double t_pc,
                                                                    std::vector<double> t_checkpoints,
                                                                    double t_abs_err = 1.0e-5,
                                                                    double t_rel_err = 1.0e-5,
                                                                    double t_init_step = 1.0e-6)
{
    /*
    Input: filename of the EOS, central pressure, number of checkpoints pressure, the checkpoint array
    Return: mass, radius, lambda, mass in checkpoins and radius in checkpoints
    */

    state_type state{t_pc*MEVFM3/TOPA, 0, 2}; // initial y is always 2
    auto eos = EOSFromFile(t_EOS_filename);
    TOV_eq<SEOS> tov(eos);

    list mass, radius;
    double R;
    CheckpointState observer(t_checkpoints, mass, radius, R);

    try
    {
        using namespace boost::numeric::odeint;
        integrate_adaptive( make_controlled<error_stepper_type>( t_abs_err , t_rel_err ) , 
                            tov , 
                            state , 
                            1e-5 ,    // initial radius (cannot be 0 as it is singular there 
                            200.0 ,   // final radius (anything that is ridicuously large will work. This will break when surface is reached
                            t_init_step,  // initial step size (for reference only. It will be adaptively changed
                            observer );
    }
    catch( const std::invalid_argument& e)
    {}


    double r = R/TOKM;
    double k2 = tov.GetK2(state, r);
    double lambda = 2*k2*pow(r*TOKM*1e3, 5)/(3*G);
    double dimlambda = lambda/pow(G, 4)/pow(state[1]*MODOT, 5)*pow(C, 10);

    return std::make_tuple(state[1], R, dimlambda, mass, radius);
}

std::tuple<list, list, list> TidalLove_analysis(const std::string& t_EOS_filename,
                                                double t_pc)
{
    state_type state{t_pc*MEVFM3/TOPA, 0, 2}; // initial y is always 2
    auto eos = EOSFromFile(t_EOS_filename);
    TOV_eq<SEOS> tov(eos);

    list mass, radius, pressure;
    CheckpointSaveAll observer(mass, radius, pressure);
    try
    {
        using namespace boost::numeric::odeint;
        integrate_adaptive( make_controlled<error_stepper_type>( 1.0e-10 , 1.0e-7 ) , 
                            tov , 
                            state , 
                            1e-5 ,    // initial radius (cannot be 0 as it is singular there 
                            200.0 ,   // final radius (anything that is ridicuously large will work. This will break when surface is reached
                            0.00001,  // initial step size (for reference only. It will be adaptively changed
                            observer );
    }
    catch( const std::invalid_argument& e)
    {}

    return std::make_tuple(mass, radius, pressure);
}

p::tuple wrap_TidalLove_analysis(const std::string& t_EOS_filename, double t_pc)
{
    auto result = TidalLove_analysis(t_EOS_filename, t_pc);
    auto mass = std::get<0>(result);
    auto radius = std::get<1>(result);
    auto pressure = std::get<2>(result);
    return p::make_tuple(wrap_to_ndarray(mass), wrap_to_ndarray(radius), wrap_to_ndarray(pressure));
}

p::tuple wrap_TidalLove_individual(const std::string& t_EOS_filename,
                                   double t_pc, 
                                   np::ndarray const & array,
                                   double t_abs_err = 1.0e-5,
                                   double t_rel_err = 1.0e-5,
                                   double t_init_step = 1.0e-5)
{
    
    auto checkpoint = wrap_from_ndarray(array);
    auto result = TidalLove_individual(t_EOS_filename, t_pc, checkpoint, t_abs_err, t_rel_err, t_init_step);
    
    // convert mass and radius into python array
    auto M = std::get<0>(result);
    auto R = std::get<1>(result);
    auto dimlambda = std::get<2>(result);
    auto mass = std::get<3>(result);
    auto radius = std::get<4>(result);

    return p::make_tuple(M, R, dimlambda, wrap_to_ndarray(mass), wrap_to_ndarray(radius));
}
 
BOOST_PYTHON_FUNCTION_OVERLOADS(wrap_TidalLove_individual_overloads, wrap_TidalLove_individual, 3, 6)

BOOST_PYTHON_MODULE(TidalLove_CPP)
{
    np::initialize(); // have to put this in any module that uses Boost.NumPy
    p::def("tidallove_individual", wrap_TidalLove_individual, wrap_TidalLove_individual_overloads());
    p::def("tidallove_analysis", wrap_TidalLove_analysis);
}                                         

/*int main(int argv, char** argc)
{
    PyImport_AppendInittab("TidalLove_individual", &initTidalLove_individual);

    Py_Initialize();

    PyRun_SimpleString(
        "import TidalLove_individual\n"
        "import numpy as np\n"
        "checkpoint = np.array([100., 10.])\n"
        "M, R, lambda_, mass, radius = TidalLove_individual.TidalLove_individual('EOS_10.csv', 128, 2, checkpoint)\n"
        "print(M, R, lambda_, mass, radius)\n"
    );
    Py_Finalize();
    return 0;    
}*/

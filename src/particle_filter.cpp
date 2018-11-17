/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    /* Get deviations */
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    /* Create normal distribution for x */
    normal_distribution<double> dist_x( 0, std_x );
    normal_distribution<double> dist_y( 0, std_y );
    normal_distribution<double> dist_theta( 0, std_theta );

    for( int i=0; i < num_particles; ++i ) {
        Particle p;
        p.id = i;
        p.x = x + dist_x(gen);
        p.y = y + dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;

        particles.push_back( p );
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    /* Create normal distribution for sensor measurements */
    normal_distribution<double> dist_x( 0, std_pos[0] );
    normal_distribution<double> dist_y( 0, std_pos[1] );
    normal_distribution<double> dist_theta( 0, std_pos[2] );

    for ( auto &p: particles ) {

        if ( fabs( yaw_rate ) >= 0.001  ) {
            p.x += (velocity/yaw_rate)*( sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
            p.y += (velocity/yaw_rate)*( cos(p.theta ) - cos(p.theta + yaw_rate*delta_t));
            p.theta += yaw_rate*delta_t;
        } else {
            p.x = dist_x(gen);
            p.y = dist_y(gen);
            p.theta += dist_theta(gen);

        }
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    /* Get all observations */
    for ( auto &obs: observations ) {

        auto min_distance = std::numeric_limits<double>::max();
        int min_pred_id = 0;

        /* Get all predicted */
        for ( auto &pred: predicted ) {
            /* Get distance between prediction and observation */
            auto cur_dist = dist(pred.x, pred.y,obs.x, obs.y);
            if ( cur_dist < min_distance ) {
                min_distance = cur_dist;
                min_pred_id = pred.id;
            }
        }

        obs.id = min_pred_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    /* Get all particles */
    for ( auto &p: particles ) {

        std::vector<LandmarkObs> predicts;

        /* Get all landmark */
        for ( auto &lnd: map_landmarks.landmark_list ) {

            /* Calculate distance between objects */
            auto dst = dist(p.x, p.y, lnd.x_f, lnd.y_f);
            if ( dst <= sensor_range ) {
                LandmarkObs l = {lnd.id_i,lnd.x_f, lnd.y_f};
                predicts.push_back(l);
            }
        }

        /* Transform observations in MAP coondination sysstem */

        vector<LandmarkObs> transformed_obs; /* result vector */
        for ( auto &obs: observations ) {
            double n_x = cos(p.theta)*obs.x - sin(p.theta)*obs.y + p.x;
            double n_y = sin(p.theta)*obs.x + cos(p.theta)*obs.y + p.y;
            LandmarkObs l = {obs.id, n_x, n_y };
            transformed_obs.push_back(l);
        }

        dataAssociation( predicts, transformed_obs );

        p.weight = 1;

        for ( auto &tr_obs: transformed_obs ) {

            double pr_x, pr_y;

            for( auto &pred: predicts) {
                if ( tr_obs.id == pred.id ) {
                    pr_x = pred.x;
                    pr_y = pred.y;
                    break;
                }
            }

            /* Calculate weight */
            double l_x = std_landmark[0];
            double l_y = std_landmark[1];
            double w = ( 1/(2*M_PI*l_x*l_y)) * exp( -( pow(pr_x-tr_obs.x,2)/(2*pow(l_x, 2)) + (pow(pr_y-tr_obs.y,2)/(2*pow(l_y, 2))) ) );

            p.weight *= w;

        }

    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<Particle> new_particles;

    std::vector<double> weights;
    auto max_weight = std::numeric_limits<double>::min();
    for ( auto &p: particles ) {
        weights.push_back(p.weight);
        if ( p.weight > max_weight ) max_weight = p.weight;
    }

    uniform_int_distribution<int> sample_idx(0,num_particles-1);
    auto idx = sample_idx(gen);

    uniform_real_distribution<double> sample_weight(0, max_weight);

    double beta = 0.0;

    // spin the resample wheel!
    for (auto &p: particles) {
        beta += sample_idx(gen) * 2.0;
        while (beta > weights[idx]) {
          beta -= weights[idx];
          idx = (idx + 1) % num_particles;
        }
        new_particles.push_back(particles[idx]);
      }

    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

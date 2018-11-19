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
        p.x = x;

        p.y = y;
        p.theta = dist_theta(gen);
        p.weight = 1;

        p.x += dist_x(gen);
        p.y += dist_y(gen);

        particles.emplace_back( p );
        weights.emplace_back(p.weight);
        printParticle(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	/*
	 * Add measurements to each particle and add random Gaussian noise.
	 */
    step++;

    /* Create normal distribution for sensor measurements */
    normal_distribution<double> dist_x( 0, std_pos[0] );
    normal_distribution<double> dist_y( 0, std_pos[1] );
    normal_distribution<double> dist_theta( 0, std_pos[2] );

    for ( auto &p: particles ) {

        if ( fabs( yaw_rate ) >= 0.0001  ) {
            p.x += (velocity/yaw_rate)*( sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
            p.y += (velocity/yaw_rate)*( cos(p.theta ) - cos(p.theta + yaw_rate*delta_t));
            p.theta += yaw_rate*delta_t;
        } else {
            p.x += velocity*cos(p.theta)*delta_t;
            p.y += velocity*sin(p.theta)*delta_t;

        }

        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);

        if ( is_debug )
            printParticle(p);

    }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs> &predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	/*
	 * Find the predicted measurement that is closest to each observed measurement and assign the
	 * observed measurement to this particular landmark.
	 */

    double dist_limit = sensor_range/2;

    /* Get all observations */
    for (auto &o: observations) {

        // init minimum distance to maximum possible
        double min_dist = dist_limit;

        // init id of landmark from map placeholder
        int map_id = -1;

        for ( auto &p: predicted ) {
          // get distance between landmarks
          double cur_dist = dist(o.x, o.y, p.x, p.y);

          // find nearest landmark
          if (cur_dist < min_dist) {
            min_dist = cur_dist;
            map_id = p.id;
          }
        }

        // set the observation's id to the nearest predicted landmark's id
        if ( map_id >= 0 ) {
           o.id = map_id;
           if (is_debug ) printf("Associate map_id %d; min_dist: %f\n", map_id, min_dist);
        }
      }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	/*
	 * Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	 * more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	 * NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	 * according to the MAP'S coordinate system. You will need to transform between the two systems.
	 * Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	 * The following is a good resource for the theory:
	 * https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	 * and the following is a good resource for the actual equation to implement (look at equation
	 * 3.33
	 * http://planning.cs.uiuc.edu/node99.html
	 */

    double weights_sum = 0;
    /* Get all particles */
    for ( int i=0; i < num_particles; ++i) {

        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        std::vector<LandmarkObs> predicts;

        /* Get all landmark */
        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
            /*
             * To increase performance we had to change `dist` function to simple  comparison difference in x and y coordinates
             */
            if ((fabs((p_x - current_landmark.x_f)) <= sensor_range) && (fabs((p_y - current_landmark.y_f)) <= sensor_range)) {
                predicts.emplace_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
            }
        }
        if (is_debug) printf("Landmark count: %ld\n", predicts.size());
        /* Transform observations in MAP coondination sysstem */

        vector<LandmarkObs> transformed_obs; /* result vector */
        for (unsigned int j = 0; j < observations.size(); ++j) {
            auto obs = observations[j];
            double n_x = cos(p_theta)*obs.x - sin(p_theta)*obs.y + p_x;
            double n_y = sin(p_theta)*obs.x + cos(p_theta)*obs.y + p_y;
            LandmarkObs l = {obs.id, n_x, n_y };
            transformed_obs.emplace_back(l);
        }

        dataAssociation( predicts, transformed_obs, sensor_range );

        particles[i].weight = 1;

        /*Calculate the weight of particle based on the multivariate Gaussian probability function*/
        for (unsigned int k = 0; k < transformed_obs.size(); k++) {
            double trans_obs_x = transformed_obs[k].x;
            double trans_obs_y = transformed_obs[k].y;
            int trans_obs_id = transformed_obs[k].id;
            double multi_prob = 1.0;

            for (unsigned int l = 0; l < predicts.size(); l++) {
                double pred_landmark_x = predicts[l].x;
                double pred_landmark_y = predicts[l].y;
                int pred_landmark_id = predicts[l].id;

                if (trans_obs_id == pred_landmark_id) {
                    double sigma_x = std_landmark[0];
                    double sigma_y = std_landmark[1];
                    double sigma_x_2 = pow(sigma_x, 2);
                    double sigma_y_2 = pow(sigma_y, 2);
                    double gauss_norm = (1.0/(2.0 * M_PI * sigma_x * sigma_y));

                    auto distance_x_2 = (trans_obs_x - pred_landmark_x) * ( trans_obs_x - pred_landmark_x);
                    auto distance_y_2 = (trans_obs_y - pred_landmark_y) * ( trans_obs_y - pred_landmark_y);
                    auto exponent = distance_x_2/(2.0 * sigma_x_2) + distance_y_2/(2.0 * sigma_y_2);

                    if (is_debug) printf("distance: %f %f; exp: %f\n", distance_x_2, distance_y_2, exponent );

                    multi_prob = gauss_norm * exp(-exponent);

                    if (is_debug) printf("Particle %d for landmark %d has weight %f %f\n", i, trans_obs_id, multi_prob, particles[i].weight );

                    particles[i].weight = particles[i].weight*multi_prob;

                    if (is_debug) printf("Result weight for particle %d: %f\n", i, particles[i].weight );

                    break;
                }
            }
        }
        weights_sum += particles[i].weight;
    }

    if (is_debug) printf("Sum of weights: %f\n", weights_sum);

    for (unsigned int i = 0; i < particles.size(); i++) {
      particles[i].weight /= weights_sum;
      weights[i] = particles[i].weight;
      if (is_debug) printf("Particle %d has norm weight: %f\n", i, weights[i] );
    }


}

void ParticleFilter::resample() {

	/*
	 * Resample particles with replacement with probability proportional to their weight.
	 */

    vector<Particle> new_particles;

    auto max_weight = std::numeric_limits<double>::min();
    for ( auto &w: weights ) {
        if ( w > max_weight ) max_weight = w;
    }

    if (is_debug) printf("Max weight: %f\n", max_weight);

    uniform_int_distribution<int> sample_idx(0,num_particles-1);
    auto idx = sample_idx(gen);

    uniform_real_distribution<double> sample_weight(0, max_weight);

    double beta = 0.0;

    // spin the resample wheel!
    for (int i =0; i < num_particles; ++i ) {
        beta += sample_weight(gen) * 2.0;
        while (beta > weights[idx]) {
          beta -= weights[idx];
          idx = (idx + 1) % num_particles;
        }
        new_particles.emplace_back(particles[idx]);
      }

    particles = new_particles;

    if (is_debug ) {
        for ( auto &p: particles ) {
            printParticle(p);
        }
    }

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

    return particle;
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

void ParticleFilter::printParticle( const Particle &p ) {
    printf("particle %d; x: %f; y: %f; theta: %f; weight: %f\n", p.id, p.x, p.y, p.theta, p.weight );
}

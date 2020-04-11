%
% superclass for all superpositions of
% randomly-apodized
% quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2020-04-08
% modified: 2020-04-08
%
classdef ra < scattering.sequences.syntheses.random.random

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        setting_rng ( 1, 1 ) auxiliary.setting_rng      % settings of the random number generator

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = ra( settings_rng )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure class auxiliary.setting_rng for settings_rng

            %--------------------------------------------------------------
            % 2.) create superpositions of randomly-apodized quasi-(d-1)-spherical waves
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.syntheses.random.random( size( settings_rng ) );

            % iterate superpositions
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).setting_rng = settings_rng( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = ra( settings_rng )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute time delays and apodization weights (scalar)
        %------------------------------------------------------------------
        function [ time_delays, apodization_weights, indices_active ] = compute_delays_scalar( RA, xdc_array, ~ )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.syntheses.wave (scalar) for RA
            % calling function ensures class scattering.sequences.setups.transducers.array (scalar) for xdc_array
            % calling function ensures class physical_values.meter_per_second (scalar) for c_avg

            %--------------------------------------------------------------
            % 2.) compute time delays and apodization weights (scalar)
            %--------------------------------------------------------------
            % seed random number generator
            rng( RA.setting_rng.seed, RA.setting_rng.str_name );

            % sample Bernoulli distribution
            apodization_weights = rand( 1, xdc_array.N_elements );
            indicator = ( apodization_weights >= 0.5 );
            apodization_weights( indicator ) = 1;
            apodization_weights( ~indicator ) = -1;

            % b) time delays
            time_delays = physical_values.second( zeros( 1, xdc_array.N_elements ) );

            % c) all array elements are active
            indices_active = (1:xdc_array.N_elements);

        end % function [ time_delays, apodization_weights, indices_active ] = compute_delays_scalar( RA, xdc_array, ~ )

	end % methods (Access = protected, Hidden)

end % classdef ra < scattering.sequences.syntheses.random.random

%
% superclass for all superpositions of
% randomly-apodized and randomly-delayed
% quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2020-04-08
% modified: 2020-08-01
%
classdef rad < scattering.sequences.syntheses.random.random

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        setting_rng_apo ( 1, 1 ) auxiliary.setting_rng      % settings of the random number generator
        setting_rng_del ( 1, 1 ) auxiliary.setting_rng      % settings of the random number generator
        e_theta ( 1, 1 ) math.unit_vector                   % preferred direction of propagation for permutation of delays

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = rad( settings_rng_apo, settings_rng_del, e_theta )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure class auxiliary.setting_rng for settings_rng_apo
            % property validation functions ensure class auxiliary.setting_rng for setting_rng_del
            % property validation functions ensure class math.unit_vector for e_theta

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( settings_rng_apo, settings_rng_del, e_theta );

            %--------------------------------------------------------------
            % 2.) create superpositions of randomly-apodized and randomly-delayed quasi-(d-1)-spherical waves
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.syntheses.random.random( size( settings_rng_apo ) );

            % iterate superpositions
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).setting_rng_apo = settings_rng_apo( index_object );
                objects( index_object ).setting_rng_del = settings_rng_del( index_object );
                objects( index_object ).e_theta = e_theta( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = rad( settings_rng_apo, setting_rng_del, e_theta )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute time delays and apodization weights (scalar)
        %------------------------------------------------------------------
        function [ time_delays, apodization_weights, indices_active ] = compute_delays_scalar( RAD, xdc_array, c_avg )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.syntheses.wave (scalar) for RAD
            % calling function ensures class scattering.sequences.setups.transducers.array (scalar) for xdc_array
            % calling function ensures class physical_values.meter_per_second (scalar) for c_avg

            % ensure matching numbers of dimensions
            if numel( RAD.e_theta.components ) - xdc_array.N_dimensions ~= 1
                errorStruct.message = 'Number of components in RAD.e_theta must equal number of dimensions of xdc_array minus 1!';
                errorStruct.identifier = 'compute_delays_scalar:DimensionMismatch';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute time delays and apodization weights (scalar)
            %--------------------------------------------------------------
            % seed random number generator
            rng( RAD.setting_rng_apo.seed, RAD.setting_rng_apo.str_name );

            % sample Bernoulli distribution
            apodization_weights = rand( xdc_array.N_elements, 1 );
            indicator = ( apodization_weights >= 0.5 );
            apodization_weights( indicator ) = 1;
            apodization_weights( ~indicator ) = -1;

            % seed random number generator
            rng( RAD.setting_rng_del.seed, RAD.setting_rng_del.str_name );

            % exclude lateral dimensions with less than two array elements
            indicator_dimensions = xdc_array.N_elements_axis > 1;

            % compute permissible maximum time shift
            t_shift_max = sum( ( xdc_array.N_elements_axis( indicator_dimensions )' - 1 ) .* xdc_array.cell_ref.edge_lengths( indicator_dimensions ) .* abs( RAD.e_theta.components( [ indicator_dimensions; false ] ) ), 2 ) / c_avg;
            % incorrect value for reproduction of old results: T_inc = t_shift_max / xdc_array.N_elements;
            T_inc = t_shift_max / ( xdc_array.N_elements - 1 );

            % compute random time delays
            time_delays = ( randperm( xdc_array.N_elements ).' - 1 ) * T_inc;

            % c) all array elements are active
            indices_active = (1:xdc_array.N_elements).';

        end % function [ time_delays, apodization_weights, indices_active ] = compute_delays_scalar( RAD, xdc_array, c_avg )

	end % methods (Access = protected, Hidden)

end % classdef rad < scattering.sequences.syntheses.random.random

%
% superclass for all quasi-(d-1)-spherical waves with
% virtual sources
%
% author: Martin F. Schiffner
% date: 2020-04-08
% modified: 2020-08-01
%
classdef qsw < scattering.sequences.syntheses.deterministic.deterministic

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        position_src ( 1, : ) physical_values.length	% position of the virtual source
        angles ( 1, : ) double                          % aperture angles (rad); 0 < angles < pi

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = qsw( positions_src, angles )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % property validation functions ensure class physical_values.length for positions_src
            % property validation functions ensure double for angles

            %--------------------------------------------------------------
            % 2.) create quasi-spherical waves (QSWs) with virtual sources
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.syntheses.deterministic.deterministic( [ size( positions_src, 1 ), 1 ] );

            % iterate QSWs
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).position_src = positions_src( index_object, : );
                objects( index_object ).angles = angles( index_object, : );

            end % for index_object = 1:numel( objects )

        end % function objects = qsw( positions_src, angles )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute time delays and apodization weights (scalar)
        %------------------------------------------------------------------
        function [ time_delays, apodization_weights, indices_active ] = compute_delays_scalar( QSW, xdc_array, c_avg )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.syntheses.wave (scalar) for QSW
            % calling function ensures class scattering.sequences.setups.transducers.array (scalar) for xdc_array
            % calling function ensures class physical_values.meter_per_second (scalar) for c_avg

            %--------------------------------------------------------------
            % 2.) determine active array elements
            %--------------------------------------------------------------
            % a) compute distances from virtual source to physical sources
            vectors_src_ctr = [ xdc_array.positions_ctr, zeros( xdc_array.N_elements, 1 ) ] - QSW.position_src;
            distances_src_ctr = vecnorm( vectors_src_ctr, 2, 2 );
            indicator_distance = double( distances_src_ctr ) >= eps;

            % b) determine active array elements
            indicator_active = false( xdc_array.N_elements, 1 );
            indicator_active( ~indicator_distance ) = true;
            indicator_active( indicator_distance ) = all( asin( abs( vectors_src_ctr( indicator_distance, 1:xdc_array.N_dimensions ) ./ distances_src_ctr( indicator_distance ) ) ) <= QSW.angles / 2, 2 );
            indices_active = find( indicator_active );

            % ensure at least one active element
            if ~( numel( indices_active ) > 0 )
                errorStruct.message = 'QSW requires at least one active array element!';
                errorStruct.identifier = 'compute_delays_scalar:NoActiveElement';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 3.) compute time delays and apodization weights
            %--------------------------------------------------------------
            % a) unity apodization weights
            apodization_weights = ones( numel( indices_active ), 1 );

            % b) time delays
            time_delays = distances_src_ctr( indicator_active ) / c_avg;
            time_delays = time_delays - min( time_delays );

        end % function [ time_delays, apodization_weights, indices_active ] = compute_delays_scalar( QSW, xdc_array, c_avg )

	end % methods (Access = protected, Hidden)

end % classdef qsw < scattering.sequences.syntheses.deterministic.deterministic

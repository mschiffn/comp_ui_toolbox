%
% superclass for all steered quasi-plane waves (QPWs)
%
% author: Martin F. Schiffner
% date: 2020-04-07
% modified: 2020-04-12
%
classdef qpw < scattering.sequences.syntheses.deterministic.deterministic

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        e_theta ( 1, 1 ) math.unit_vector { mustBeNonempty } = math.unit_vector % preferred direction of propagation
%         T_clk ( 1, 1 ) physical_values.time

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = qpw( e_theta )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty e_theta
            if nargin < 1 || isempty( e_theta )
                e_theta = math.unit_vector( [ 1, 0, 0 ] );
            end

            % property validation functions ensure class math.unit_vector for e_theta

            %--------------------------------------------------------------
            % 2.) create steered quasi-plane waves (QPWs)
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.syntheses.deterministic.deterministic( size( e_theta ) );

            % iterate QPWs
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).e_theta = e_theta( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = qpw( e_theta )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute time delays and apodization weights (scalar)
        %------------------------------------------------------------------
        function [ time_delays, apodization_weights, indices_active ] = compute_delays_scalar( QPW, xdc_array, c_avg )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.syntheses.wave (scalar) for QPW
            % calling function ensures class scattering.sequences.setups.transducers.array (scalar) for xdc_array
            % calling function ensures class physical_values.meter_per_second (scalar) for c_avg

            % ensure matching numbers of dimensions
            if numel( QPW.e_theta.components ) - xdc_array.N_dimensions ~= 1
                errorStruct.message = 'Number of components in QPW.e_theta must equal number of dimensions of xdc_array minus 1!';
                errorStruct.identifier = 'compute_delays_scalar:DimensionMismatch';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute time delays and apodization weights (scalar)
            %--------------------------------------------------------------
            % a) unity apodization weights
            apodization_weights = ones( xdc_array.N_elements, 1 );

            % b) time delays
            time_delays = xdc_array.positions_ctr * QPW.e_theta.components( 1:(end - 1) ).' / c_avg;
            time_delays = time_delays - min( time_delays );

            % c) all array elements are active
            indices_active = (1:xdc_array.N_elements).';

        end % function [ time_delays, apodization_weights, indices_active ] = compute_delays_scalar( QPW, xdc_array, c_avg )

	end % methods (Access = protected, Hidden)

end % classdef qpw < scattering.sequences.syntheses.deterministic.deterministic

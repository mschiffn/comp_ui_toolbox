%
% abstract superclass for all incident waves
%
% author: Martin F. Schiffner
% date: 2020-04-08
% modified: 2020-04-08
%
classdef (Abstract) wave

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = wave( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'wave:NoRowVector';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBePositive( size );
            mustBeInteger( size );
            mustBeNonempty( size );

            %--------------------------------------------------------------
            % 2.) create regularization options
            %--------------------------------------------------------------
            % repeat default regularization options
            objects = repmat( objects, size );

        end % function objects = wave( size )

        %------------------------------------------------------------------
        % compute time delays and apodization weights
        %------------------------------------------------------------------
        function [ time_delays, apodization_weights, indices_active ] = compute_delays( waves, xdc_arrays, c_avg )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.syntheses.wave
            if ~isa( waves, 'scattering.sequences.syntheses.wave' )
                errorStruct.message = 'waves must be scattering.sequences.syntheses.wave!';
                errorStruct.identifier = 'compute_delays:NoWaves';
                error( errorStruct );
            end

            % ensure class scattering.sequences.setups.transducers.array
            if ~isa( xdc_arrays, 'scattering.sequences.setups.transducers.array' )
                errorStruct.message = 'xdc_arrays must be scattering.sequences.setups.transducers.array!';
                errorStruct.identifier = 'compute_delays:NoArrays';
                error( errorStruct );
            end

            % ensure class physical_values.meter_per_second
            if ~isa( c_avg, 'physical_values.meter_per_second' )
                errorStruct.message = 'c_avg must be physical_values.meter_per_second!';
                errorStruct.identifier = 'compute_delays:NoSoundSpeeds';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            [ waves, xdc_arrays, c_avg ] = auxiliary.ensureEqualSize( waves, xdc_arrays, c_avg );

            %--------------------------------------------------------------
            % 2.) compute time delays and apodization weights
            %--------------------------------------------------------------
            % specify cell arrays
            time_delays = cell( size( waves ) );
            apodization_weights = cell( size( waves ) );
            indices_active = cell( size( waves ) );

            % iterate incident waves
            for index_object = 1:numel( waves )

                % compute time delays and apodization weights (scalar)
                [ time_delays{ index_object }, apodization_weights{ index_object }, indices_active{ index_object } ] = compute_delays_scalar( waves( index_object ), xdc_arrays( index_object ), c_avg( index_object ) );

            end % for index_object = 1:numel( waves )

            % avoid cell arrays for single waves
            if isscalar( waves )
                time_delays = time_delays{ 1 };
                apodization_weights = apodization_weights{ 1 };
                indices_active = indices_active{ 1 };
            end

        end % function [ time_delays, apodization_weights, indices_active ] = compute_delays( waves, xdc_arrays, c_avg )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute time delays and apodization weights (scalar)
        %------------------------------------------------------------------
        [ time_delays, apodization_weights, indices_active ] = compute_delays_scalar( wave, xdc_array, c_avg )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) wave

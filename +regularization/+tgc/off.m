%
% superclass for all inactive time gain compensation (TGC) options
%
% author: Martin F. Schiffner
% date: 2019-12-15
% modified: 2020-02-21
%
classdef off < regularization.tgc.tgc

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = off( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers

            %--------------------------------------------------------------
            % 2.) create inactive TGC options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.tgc.tgc( size );

        end % function objects = off( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( tgcs_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.tgc.off
            if ~isa( tgcs_off, 'regularization.tgc.off' )
                errorStruct.message = 'tgcs_off must be regularization.tgc.off!';
                errorStruct.identifier = 'string:NoTGCsOff';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "off"
            strs_out = repmat( "off", size( tgcs_off ) );

        end % function strs_out = string( tgcs_off )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % create linear transform (scalar)
        %------------------------------------------------------------------
        function [ LT, LTs_measurement ] = get_LT_scalar( ~, operator_born )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class regularization.tgc.tgc (scalar)
            % calling function ensures class scattering.operator_born (scalar)

            %--------------------------------------------------------------
            % 2.) create linear transform (scalar)
            %--------------------------------------------------------------
            % numbers of observations for all sequential pulse-echo measurements
            N_observations_mix = { operator_born.sequence.settings( operator_born.indices_measurement_sel ).N_observations };
            N_observations_measurement = cellfun( @( x ) sum( x( : ) ), N_observations_mix );

            % specify cell array for LTs_measurement
            LTs_measurement = cell( numel( operator_born.indices_measurement_sel ), 1 );

            % iterate selected sequential pulse-echo measurements
            for index_measurement_sel = 1:numel( operator_born.indices_measurement_sel )

                % create identity for the selected sequential pulse-echo measurement
                LTs_measurement{ index_measurement_sel } = linear_transforms.identity( N_observations_measurement( index_measurement_sel ) );

            end

            % create identity for all selected sequential pulse-echo measurements
            LT = linear_transforms.identity( sum( N_observations_measurement( : ) ) );

        end % function [ LT, LTs_measurement ] = get_LT_scalar( ~, operator_born )

	end % methods (Access = protected, Hidden)

end % classdef off < regularization.tgc.tgc

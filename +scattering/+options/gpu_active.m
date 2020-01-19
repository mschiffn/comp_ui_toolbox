%
% superclass for all active GPU options
%
% author: Martin F. Schiffner
% date: 2019-07-29
% modified: 2020-01-18
%
classdef gpu_active < scattering.options.gpu

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        index ( 1, 1 ) uint8 { mustBeNonnegative, mustBeNonempty } = 0;	% index of GPU device

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = gpu_active( indices )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid indices

            %--------------------------------------------------------------
            % 2.) create active GPU options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.gpu( size( indices ) );

            % iterate active GPU options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).index = indices( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = gpu_active( indices )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( gpus_active )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.gpu_active
            if ~isa( gpus_active, 'scattering.options.gpu_active' )
                errorStruct.message = 'gpus_active must be scattering.options.gpu_active!';
                errorStruct.identifier = 'string:NoOptionsGPUActive';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "off"
            strs_out = repmat( "", size( gpus_active ) );

            % iterate active GPU options
            for index_object = 1:numel( gpus_active )

                strs_out( index_object ) = sprintf( "%s (device: %d)", 'active', gpus_active( index_object ).index );

            end % for index_object = 1:numel( gpus_active )

        end % function strs_out = string( gpus_active )

	end % methods

end % classdef gpu_active < scattering.options.gpu

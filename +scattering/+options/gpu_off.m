%
% superclass for all inactive GPU options
%
% author: Martin F. Schiffner
% date: 2019-07-29
% modified: 2020-01-18
%
classdef gpu_off < scattering.options.gpu

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = gpu_off( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vectors for size
            % superclass ensures positive integers for size

            %--------------------------------------------------------------
            % 2.) create inactive GPU options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.gpu( size );

        end % function objects = gpu_off( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( gpus_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.gpu_off
            if ~isa( gpus_off, 'scattering.options.gpu_off' )
                errorStruct.message = 'gpus_off must be scattering.options.gpu_off!';
                errorStruct.identifier = 'string:NoOptionsGPUOff';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "off"
            strs_out = repmat( "off", size( gpus_off ) );

        end % function strs_out = string( gpus_off )

	end % methods

end % classdef gpu_off < scattering.options.gpu

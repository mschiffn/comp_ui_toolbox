%
% superclass for all inactive GPU options
%
% author: Martin F. Schiffner
% date: 2019-07-29
% modified: 2019-07-29
%
classdef options_gpu_off < scattering.options_gpu

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_gpu_off( varargin )

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
            objects@scattering.options_gpu( size );

        end % function objects = options_gpu_off( varargin )

	end % methods

end % classdef options_gpu_off < scattering.options_gpu

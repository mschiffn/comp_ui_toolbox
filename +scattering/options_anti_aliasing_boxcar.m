%
% superclass for all boxcar spatial anti-aliasing filter options
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2019-07-31
%
classdef options_anti_aliasing_boxcar < scattering.options_anti_aliasing

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_anti_aliasing_boxcar( varargin )

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
            % 2.) create boxcar spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options_anti_aliasing( size );

        end % function objects = options_anti_aliasing_boxcar( varargin )

	end % methods

end % classdef options_anti_aliasing_boxcar < scattering.options_anti_aliasing

%
% superclass for all direct algorithm options
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2019-07-30
%
classdef options_algorithm_direct < scattering.options_algorithm

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_algorithm_direct( varargin )

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
            % 2.) create direct algorithm options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options_algorithm( size );

        end % function objects = options_algorithm_direct( varargin )

	end % methods

end % classdef options_algorithm_direct < scattering.options_algorithm

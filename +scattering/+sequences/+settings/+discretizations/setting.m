%
% superclass for all setting-based spectral discretization options
% (common frequency axis for each sequential pulse-echo measurement)
%
% author: Martin F. Schiffner
% date: 2019-08-01
% modified: 2019-10-21
%
classdef setting < scattering.sequences.settings.discretizations.options

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting( varargin )

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
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create setting-based spectral discretization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.settings.discretizations.options( size );

        end % function objects = setting( varargin )

    end % methods

end % classdef setting < scattering.sequences.settings.discretizations.options

%
% superclass for all scattering operator options
%
% author: Martin F. Schiffner
% date: 2019-02-15
% modified: 2019-03-19
%
classdef options

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        discretization ( 1, 1 ) discretizations.options = discretizations.options % spatiospectral discretization
        spatial_aliasing

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = options( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin ~= 1
                return;
            end

            %--------------------------------------------------------------
            % 2.) parse arguments
            %--------------------------------------------------------------
            for index_arg = 1:nargin

                if isa( varargin{ index_arg }, 'discretizations.options' )
                    object.discretization = varargin{ index_arg };
                end

            end % for index_arg = 1:nargin

        end % function object = options( varargin )

	end % methods

end % classdef options

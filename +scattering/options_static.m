%
% superclass for all static scattering operator options
%
% author: Martin F. Schiffner
% date: 2019-07-09
% modified: 2019-07-09
%
classdef options_static

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        discretization ( 1, 1 ) discretizations.options = discretizations.options                       % spatiospectral discretization
        materials ( 1, 1 ) scattering.options_material = scattering.options_material.compressibility	% material parameters

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = options_static( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 2.) parse arguments
            %--------------------------------------------------------------
            for index_arg = 1:nargin

                switch class( varargin{ index_arg } )

                    %------------------------------------------------------
                    % spatiospectral discretization options
                    %------------------------------------------------------
                    case 'discretizations.options'
                        object.discretization = varargin{ index_arg };

                    %------------------------------------------------------
                    % material parameters
                    %------------------------------------------------------
                    case 'scattering.options_material'
                        object.materials = varargin{ index_arg };

                    %------------------------------------------------------
                    % unknown
                    %------------------------------------------------------
                    otherwise
                        errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
                        errorStruct.identifier = 'options:UnknownClass';
                        error( errorStruct );

                end % switch class( varargin{ index_arg } )

            end % for index_arg = 1:nargin

        end % function object = options_static( varargin )

	end % methods

end % classdef options_static

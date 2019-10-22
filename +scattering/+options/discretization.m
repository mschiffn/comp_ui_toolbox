%
% superclass for all spatiospectral discretization options
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-10-21
%
classdef discretization

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        spatial ( 1, 1 ) scattering.sequences.setups.discretizations.options { mustBeNonempty } = scattering.sequences.setups.discretizations.options       % spatial discretization options
        spectral ( 1, 1 ) scattering.sequences.settings.discretizations.options { mustBeNonempty } = scattering.sequences.settings.discretizations.sequence	% spectral discretization options

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = discretization( spatial, spectral )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no input argument
            if nargin == 0
                return;
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( spatial, spectral );

            %--------------------------------------------------------------
            % 2.) create spatiospectral discretization options
            %--------------------------------------------------------------
            % repeat default spatiospectral discretization options
            objects = repmat( objects, size( spatial ) );

            % iterate spatiospectral discretization options
            for index_object = 1:numel( spatial )

                % set independent properties
                objects( index_object ).spatial = spatial( index_object );
                objects( index_object ).spectral = spectral( index_object );

            end % for index_object = 1:numel( spatial )

        end % function objects = discretization( spatial, spectral )

	end % methods

end % classdef discretization
